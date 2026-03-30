"""
DDIM Sampler for SRDiff
Denoising Diffusion Implicit Models - Deterministic and fast sampling
Reference: https://arxiv.org/abs/2010.02502
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np
from .base_sampler import BaseSampler


class DDIMSampler(BaseSampler):
    """DDIM sampler - deterministic fast sampling"""
    
    def __init__(
        self,
        num_inference_steps: int = 50,
        eta: float = 0.0,  # eta=0 for deterministic, eta=1 for DDPM
        **kwargs
    ):
        """
        Args:
            num_inference_steps: Number of denoising steps (can be < num_timesteps)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        """
        super().__init__(**kwargs)
        self.name = "DDIM"
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        # DDIM relies on the predicted x0 without aggressive clipping; allow
        # the residual to retain its dynamic range and clamp only at the very end.
        self.clip_denoised = False
        self.x0_clip_range = 1.3
        
        # Create DDIM timestep schedule
        self.setup_ddim_schedule()
    
    def setup_ddim_schedule(self):
        """Setup DDIM sampling schedule with fewer steps.

        Fixes:
        - Guard division when steps > T and ensure schedule includes t=0.
        - Precompute alpha/sigma on the correct device.
        """
        # Create evenly spaced timesteps, including 0
        steps = max(1, int(self.num_inference_steps))
        timesteps = np.linspace(0, self.num_timesteps - 1, steps, dtype=np.int64)
        timesteps = np.flip(timesteps).copy()  # Reverse for sampling from T to 0

        self.ddim_timesteps = timesteps
        self.ddim_timesteps_prev = np.append(timesteps[1:], 0)
        # Sanity check: ensure t=0 is included so we land at a clean sample
        assert int(self.ddim_timesteps[-1]) == 0, "DDIM schedule must include t=0 as the last step"

        # Precompute alpha values for DDIM timesteps (on device)
        self.ddim_alpha = self.alphas_cumprod[timesteps]
        self.ddim_alpha_prev = torch.cat([
            self.alphas_cumprod[timesteps[1:]],
            torch.tensor([1.0], device=self.device)
        ])

        # Precompute sigma for variance
        # sigma_t = eta * sqrt((1 - a_{t-1})/(1 - a_t)) * sqrt(1 - a_t/a_{t-1})
        self.ddim_sigma = (
            self.eta
            * torch.sqrt((1.0 - self.ddim_alpha_prev) / (1.0 - self.ddim_alpha + 1e-12))
            * torch.sqrt(torch.clamp(1.0 - self.ddim_alpha / (self.ddim_alpha_prev + 1e-12), min=0.0))
        )
    
    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        t_idx: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """DDIM sampling step
        
        Args:
            model: Denoising model
            x_t: Current noisy residual
            t: Current timestep tensor (batch size)
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            t_idx: Current timestep index in DDIM schedule
            
        Returns:
            x_t_minus_1: Denoised residual at t-1
            x_0_pred: Prediction of clean image
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Get actual timestep values for model
        timestep = torch.full((batch_size,), self.ddim_timesteps[t_idx], device=device, dtype=torch.long)
        
        # Get noise prediction from model
        noise_pred = model(x_t, timestep, cond, img_lr_up)
        
        # Get alpha values for current and previous timestep - reshape for proper broadcasting
        alpha_t = self.ddim_alpha[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        alpha_prev = self.ddim_alpha_prev[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        sigma_t = self.ddim_sigma[t_idx].reshape(1, *((1,) * (len(x_t.shape) - 1)))
        
        # Predict x_0 (clean residual) with numerical stability
        sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-12))
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-12))
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        else:
            x_0_pred = torch.clamp(x_0_pred, -self.x0_clip_range, self.x0_clip_range)
        
        # Direction pointing to x_t
        # Numerically safe direction term
        dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0))
        dir_xt = dir_coeff * noise_pred
        
        # Random noise (check if sigma_t is greater than 0)
        # Use tensor operations for numerical stability
        sigma_scalar = sigma_t.view(-1)[0]  # Safely extract scalar from reshaped tensor
        noise = torch.randn_like(x_t) if float(sigma_scalar) > 1e-8 else 0
        
        # Compute x_{t-1} with numerical stability
        sqrt_alpha_prev = torch.sqrt(torch.clamp(alpha_prev, min=1e-12))
        x_t_minus_1 = sqrt_alpha_prev * x_0_pred + dir_xt + sigma_t * noise
        
        return x_t_minus_1, x_0_pred
    
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
        """Full DDIM sampling loop
        
        Args:
            model: Denoising model
            shape: Shape of image to generate
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            return_intermediates: Whether to return intermediate samples
            use_residual: Whether to use residual diffusion
            res_rescale: Residual rescaling factor
            
        Returns:
            Dictionary containing final sample and optionally intermediates
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Initialize with noise (in residual space if using residual)
        if use_residual:
            img = torch.randn(shape, device=device)
        else:
            # For non-residual, start from noisy version of upsampled image
            # Note: DDIM doesn't support arbitrary starting timesteps like DDPM
            # So we start from fully noisy version
            start_t = len(self.ddim_timesteps) - 1
            t_start = torch.full((batch_size,), self.ddim_timesteps[start_t], device=device, dtype=torch.long)
            img = self.q_sample(img_lr_up, t_start)
        
        # Progress bar
        iterator = self.get_iterator(range(len(self.ddim_timesteps)))
        
        intermediates = []
        
        for i in iterator:
            # Get current timestep index
            t_idx = i
            
            # Create timestep tensor for API compatibility
            t_batch = torch.full((batch_size,), self.ddim_timesteps[t_idx], device=device, dtype=torch.long)
            
            # Single sampling step
            img, x_0_pred = self.sample_step(
                model=model,
                x_t=img,
                t=t_batch,
                cond=cond,
                img_lr_up=img_lr_up,
                t_idx=t_idx,
                **kwargs
            )
            
            if return_intermediates:
                # Convert residual to image for visualization
                if use_residual:
                    img_vis = self.res2img(img, img_lr_up, res_rescale)
                    x_0_vis = self.res2img(x_0_pred, img_lr_up, res_rescale) if x_0_pred is not None else None
                else:
                    img_vis = img
                    x_0_vis = x_0_pred
                    
                intermediates.append({
                    't': self.ddim_timesteps[t_idx],
                    'sample': img_vis.cpu(),
                    'pred_x0': x_0_vis.cpu() if x_0_vis is not None else None
                })
        
        # Convert final residual to image
        if use_residual:
            img = self.res2img(img, img_lr_up, res_rescale)
        # Guard numerical overflow so downstream metrics see valid range
        img = torch.clamp(img, -1.0, 1.0)

        results = {'sample': img}
        if return_intermediates:
            results['intermediates'] = intermediates

        return results
    
    def get_iterator(self, timesteps):
        """Get iterator with optional progress bar"""
        if self.use_tqdm:
            from tqdm import tqdm
            return tqdm(timesteps, desc=f'{self.name} sampling ({self.num_inference_steps} steps)')
        return timesteps
    
    def get_sampling_timesteps(self, start_timestep: Optional[int] = None) -> List[int]:
        """Override to return DDIM timesteps"""
        return list(self.ddim_timesteps)
