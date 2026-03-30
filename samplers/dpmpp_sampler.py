"""
DPM-Solver++ Sampler for SRDiff
DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models
Reference: https://arxiv.org/abs/2211.01095
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np
from .base_sampler import BaseSampler


class DPMPPSampler(BaseSampler):
    """DPM-Solver++ - High-order solver for fast diffusion sampling"""
    
    def __init__(
        self,
        num_inference_steps: int = 20,
        solver_order: int = 2,  # 1, 2, or 3
        solver_type: str = 'midpoint',  # 'midpoint' or 'heun'
        lower_order_final: bool = True,
        **kwargs
    ):
        """
        Args:
            num_inference_steps: Number of denoising steps
            solver_order: Order of the solver (1=Euler, 2=Midpoint/Heun, 3=Third-order)
            solver_type: Type of second-order solver
            lower_order_final: Use lower order solver for final steps
        """
        super().__init__(**kwargs)
        self.name = "DPM-Solver++"
        self.num_inference_steps = num_inference_steps
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        
        # Setup DPM-Solver schedule
        self.setup_dpm_schedule()
        
        # History buffer for multi-step methods
        self.noise_pred_history = []
    
    def setup_dpm_schedule(self):
        """Setup DPM-Solver++ sampling schedule.

        Important fix: ensure the schedule includes t=0 so the final step
        actually lands on a fully denoised sample. The previous implementation
        dropped t=0 (by slicing with `[:-1]`), which left the sampler stuck
        at a nonzero noise level and produced noisy outputs.
        """
        # Use evenly spaced discrete timesteps that include 0
        # e.g., for T=1000 and steps=50 -> [999, 979, ..., 0]
        timesteps = torch.linspace(0, self.num_timesteps - 1, self.num_inference_steps, dtype=torch.long)
        self.dpm_timesteps = timesteps.flip(0)
        # Sanity check: ensure the schedule ends at t=0
        assert int(self.dpm_timesteps[-1].item()) == 0, "DPM++ schedule must include t=0 as the last step"

        # Precompute alpha_bar for current and "previous" (next index in schedule) steps
        self.dpm_alpha = self.alphas_cumprod[self.dpm_timesteps]
        # For the last step (t=0), set alpha_prev=1 and sigma_prev=0
        self.dpm_alpha_prev = torch.cat([
            self.alphas_cumprod[self.dpm_timesteps[1:]],
            torch.tensor([1.0], device=self.device)
        ])

        # For reference (not strictly needed by our simplified updates):
        alpha_t = torch.sqrt(self.dpm_alpha)
        sigma_t = torch.sqrt(1.0 - self.dpm_alpha)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t + 1e-12)
        # Step sizes in log-SNR space
        self.h = lambda_t[1:] - lambda_t[:-1]
        
    def get_x_and_pred_x0(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_idx: int,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current x and predict x_0 using the model"""
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Get timestep
        t = self.dpm_timesteps[t_idx]
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Get model prediction
        noise_pred = model(x_t, t_batch, cond, img_lr_up)
        
        # Convert to x_0 prediction (x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps)
        alpha_bar_t = self.alphas_cumprod[t].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12))
        x_0_pred = (x_t - sigma_t * noise_pred) / sqrt_alpha_bar_t
        
        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
        return noise_pred, x_0_pred
    
    def dpm_solver_first_order_update(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """First-order (Euler) update"""
        # Get alpha_bar and sigma values for current and target (prev in schedule)
        alpha_bar_t = self.dpm_alpha[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        alpha_bar_s = self.dpm_alpha_prev[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))
        sigma_s = torch.sqrt(torch.clamp(1.0 - alpha_bar_s, min=1e-12))
        
        # Compute x_0 prediction
        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
        x_0_pred = (x_t - sigma_t * noise_pred) / sqrt_alpha_bar_t
        
        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Update x_t -> x_s
        sqrt_alpha_bar_s = torch.sqrt(torch.clamp(alpha_bar_s, min=1e-12))
        x_s = sqrt_alpha_bar_s * x_0_pred + sigma_s * noise_pred
        
        return x_s
    
    def dpm_solver_second_order_update(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        noise_pred_t: torch.Tensor,
        noise_pred_s: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        cond: Optional[torch.Tensor] = None,
        img_lr_up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Second-order update (Midpoint or Heun)"""
        # Get alpha_bar and sigma values for current and target steps
        alpha_bar_t = self.dpm_alpha[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        alpha_bar_s = self.dpm_alpha_prev[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))
        sigma_s = torch.sqrt(torch.clamp(1.0 - alpha_bar_s, min=1e-12))

        # Work with the actual discrete timestep values for midpoint selection
        t_value = int(self.dpm_timesteps[t_idx].item())
        next_idx = min(t_idx + 1, len(self.dpm_timesteps) - 1)
        s_value = int(self.dpm_timesteps[next_idx].item())
        if t_value == s_value:
            mid_value = t_value
        else:
            mid_value = int(round((t_value + s_value) / 2.0))
        mid_value = max(min(mid_value, t_value), s_value)

        if self.solver_type == 'midpoint':
            # Midpoint method
            # First compute midpoint using true timestep values
            alpha_bar_mid = self.alphas_cumprod[mid_value].reshape(1, *((1,) * (len(x_t.shape) - 1)))
            sigma_mid = torch.sqrt(torch.clamp(1 - alpha_bar_mid, min=1e-12))

            # x_0 prediction from t
            sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
            x_0_pred_t = (x_t - sigma_t * noise_pred_t) / sqrt_alpha_bar_t
            if self.clip_denoised:
                x_0_pred_t = torch.clamp(x_0_pred_t, -1.0, 1.0)

            # Get x at midpoint
            sqrt_alpha_bar_mid = torch.sqrt(torch.clamp(alpha_bar_mid, min=1e-12))
            x_mid = sqrt_alpha_bar_mid * x_0_pred_t + sigma_mid * noise_pred_t

            # Get noise at midpoint
            batch_size = x_t.shape[0]
            device = x_t.device
            t_mid_batch = torch.full((batch_size,), mid_value, device=device, dtype=torch.long)
            noise_pred_mid = model(x_mid, t_mid_batch, cond, img_lr_up)

            # Final update using midpoint
            x_0_pred = (x_mid - sigma_mid * noise_pred_mid) / sqrt_alpha_bar_mid
            if self.clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            sqrt_alpha_bar_s = torch.sqrt(torch.clamp(alpha_bar_s, min=1e-12))
            x_s = sqrt_alpha_bar_s * x_0_pred + sigma_s * noise_pred_mid
            
        else:  # Heun's method
            # First estimate
            sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
            x_0_pred_t = (x_t - sigma_t * noise_pred_t) / sqrt_alpha_bar_t
            if self.clip_denoised:
                x_0_pred_t = torch.clamp(x_0_pred_t, -1.0, 1.0)
            sqrt_alpha_bar_s = torch.sqrt(torch.clamp(alpha_bar_s, min=1e-12))
            x_s_est = sqrt_alpha_bar_s * x_0_pred_t + sigma_s * noise_pred_t

            # Get noise at s (if not provided)
            if noise_pred_s is None:
                batch_size = x_t.shape[0]
                device = x_t.device
                # Target time in the schedule for this step
                t_s_batch = torch.full((batch_size,), s_value, device=device, dtype=torch.long)
                noise_pred_s = model(x_s_est, t_s_batch, cond, img_lr_up)
            
            # Corrector step
            x_0_pred = (x_t - sigma_t * (noise_pred_t + noise_pred_s) / 2) / sqrt_alpha_bar_t
            if self.clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            x_s = sqrt_alpha_bar_s * x_0_pred + sigma_s * (noise_pred_t + noise_pred_s) / 2
            
        return x_s
    
    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,  # This is actually the index in DPM schedule
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        step_idx: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """DPM-Solver++ sampling step"""
        t_idx = step_idx
        
        # Get noise prediction
        noise_pred, x_0_pred = self.get_x_and_pred_x0(model, x_t, t_idx, cond, img_lr_up)
        
        # Store in history
        self.noise_pred_history.append(noise_pred)
        if len(self.noise_pred_history) > self.solver_order:
            self.noise_pred_history.pop(0)

        # Determine solver order for this step
        if self.lower_order_final and step_idx >= len(self.dpm_timesteps) - self.solver_order:
            current_order = 1
        else:
            current_order = min(self.solver_order, len(self.noise_pred_history))

        # Apply solver
        if current_order == 1:
            x_prev = self.dpm_solver_first_order_update(x_t, t_idx, noise_pred)
        elif current_order == 2:
            x_prev = self.dpm_solver_second_order_update(
                x_t, t_idx, noise_pred,
                model=model, cond=cond, img_lr_up=img_lr_up
            )
        else:  # current_order == 3
            # For simplicity, reuse second-order update
            x_prev = self.dpm_solver_second_order_update(
                x_t, t_idx, noise_pred,
                model=model, cond=cond, img_lr_up=img_lr_up
            )

        return x_prev, x_0_pred

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        return_intermediates: bool = False,
        use_residual: bool = True,
        res_rescale: float = 2.0,
        **kwargs
    ):
        """Full DPM-Solver++ sampling loop"""
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Initialize with noise (in residual space if using residual)
        if use_residual:
            img = torch.randn(shape, device=device)
        else:
            # For non-residual, start from noisy version of upsampled image
            start_t = self.dpm_timesteps[0]  # highest noise index in our schedule
            t_start = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
            img = self.q_sample(img_lr_up, t_start)

        # Reset history
        self.noise_pred_history = []

        # Progress bar
        iterator = self.get_iterator(range(len(self.dpm_timesteps)))

        intermediates: List[dict] = []
        for step_idx in iterator:
            # Single sampling step
            img, x_0_pred = self.sample_step(
                model=model,
                x_t=img,
                t=None,  # step_idx drives the schedule
                cond=cond,
                img_lr_up=img_lr_up,
                step_idx=step_idx,
                **kwargs,
            )

            if return_intermediates:
                # Convert residual to image for visualization
                if use_residual:
                    img_vis = self.res2img(img, img_lr_up, res_rescale)
                    x_0_vis = self.res2img(x_0_pred, img_lr_up, res_rescale) if x_0_pred is not None else None
                else:
                    img_vis = img
                    x_0_vis = x_0_pred

                intermediates.append(
                    {
                        't': int(self.dpm_timesteps[step_idx].item()),
                        'sample': img_vis.cpu(),
                        'pred_x0': x_0_vis.cpu() if x_0_vis is not None else None,
                    }
                )

        # Convert final residual to image
        if use_residual:
            img = self.res2img(img, img_lr_up, res_rescale)

        results = {'sample': img}
        if return_intermediates:
            results['intermediates'] = intermediates

        return results

    def get_iterator(self, timesteps):
        """Get iterator with optional progress bar"""
        if self.use_tqdm:
            from tqdm import tqdm
            return tqdm(
                timesteps, desc=f'{self.name} sampling ({self.num_inference_steps} steps, order={self.solver_order})'
            )
        return timesteps
