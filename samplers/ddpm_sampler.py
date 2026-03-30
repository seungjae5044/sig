"""
DDPM Sampler for SRDiff
Standard Denoising Diffusion Probabilistic Model sampling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base_sampler import BaseSampler


class DDPMSampler(BaseSampler):
    """DDPM sampler - standard diffusion sampling with Gaussian noise"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DDPM"
    
    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """DDPM sampling step
        
        Args:
            model: Denoising model
            x_t: Current noisy residual
            t: Current timestep
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            temperature: Noise temperature for sampling
            
        Returns:
            x_t_minus_1: Denoised residual at t-1
            x_0_pred: Prediction of clean image
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Get noise prediction from model
        noise_pred = model(x_t, t, cond, img_lr_up)
        
        # Predict x_0 from noise
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        
        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Calculate posterior mean and variance
        # q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        # Sample from posterior
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))
        
        # Add noise scaled by temperature
        x_t_minus_1 = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise * temperature
        
        return x_t_minus_1, x_0_pred
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        start_timestep: Optional[int] = None,
        return_intermediates: bool = False,
        use_residual: bool = True,
        res_rescale: float = 2.0,
        **kwargs
    ):
        """Full DDPM sampling loop with residual support
        
        Args:
            model: Denoising model
            shape: Shape of image to generate
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            start_timestep: Starting timestep
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
            if start_timestep is None:
                start_timestep = self.num_timesteps - 1
            t_start = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
            img = self.q_sample(img_lr_up, t_start)
        
        # Setup timesteps for sampling
        timesteps = self.get_sampling_timesteps(start_timestep)
        
        # Progress bar
        iterator = self.get_iterator(timesteps)
        
        intermediates = []
        
        for t in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Single sampling step
            img, x_0_pred = self.sample_step(
                model=model,
                x_t=img,
                t=t_batch,
                cond=cond,
                img_lr_up=img_lr_up,
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
                    't': t,
                    'sample': img_vis.cpu(),
                    'pred_x0': x_0_vis.cpu() if x_0_vis is not None else None
                })
        
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
            return tqdm(timesteps, desc=f'{self.name} sampling')
        return timesteps