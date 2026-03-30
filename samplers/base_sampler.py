"""
Base Sampler Abstract Class for SRDiff
Provides common interface for all diffusion samplers
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm


class BaseSampler(ABC):
    """Abstract base class for diffusion samplers"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cuda',
        clip_denoised: bool = True,
        use_tqdm: bool = True,
        precomputed_schedule: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.clip_denoised = clip_denoised
        self.use_tqdm = use_tqdm
        # Optional: use model's precomputed diffusion schedule to ensure exact match
        self._precomputed_schedule = precomputed_schedule

        # Setup diffusion parameters
        self.setup_diffusion_parameters()
        
    def setup_diffusion_parameters(self):
        """Setup alpha/beta schedules and posterior terms.

        If a precomputed schedule (from the trained GaussianDiffusion) is provided,
        use it to guarantee perfect alignment with training; otherwise, generate
        a schedule locally.
        """
        if self._precomputed_schedule is not None:
            sched = self._precomputed_schedule
            # Move to target device and cast to float32
            def todev(x):
                return x.to(self.device, dtype=torch.float32)

            self.betas = todev(sched['betas'])
            self.alphas_cumprod = todev(sched['alphas_cumprod'])
            self.alphas_cumprod_prev = todev(sched['alphas_cumprod_prev'])
            self.sqrt_alphas_cumprod = todev(sched['sqrt_alphas_cumprod'])
            self.sqrt_one_minus_alphas_cumprod = todev(sched['sqrt_one_minus_alphas_cumprod'])
            self.sqrt_recip_alphas_cumprod = todev(sched['sqrt_recip_alphas_cumprod'])
            self.sqrt_recipm1_alphas_cumprod = todev(sched['sqrt_recipm1_alphas_cumprod'])
            self.posterior_variance = todev(sched['posterior_variance'])
            self.posterior_log_variance_clipped = todev(sched['posterior_log_variance_clipped'])
            self.posterior_mean_coef1 = todev(sched['posterior_mean_coef1'])
            self.posterior_mean_coef2 = todev(sched['posterior_mean_coef2'])
            # Derive alphas for convenience
            self.alphas = 1.0 - self.betas
            # Ensure num_timesteps matches
            self.num_timesteps = int(self.betas.shape[0])
            return

        # Fallback: generate schedule locally
        if self.beta_schedule == 'linear':
            betas = np.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=np.float64)
        elif self.beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        self.alphas = torch.tensor(alphas, dtype=torch.float32, device=self.device)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=self.device)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32, device=self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32, device=self.device)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=torch.float32, device=self.device
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=torch.float32, device=self.device
        )
    
    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine beta schedule as proposed in improved DDPM"""
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, a_min=0, a_max=0.999)
    
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract values from a 1-D tensor for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process - add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and noise prediction"""
        sqrt_recip_alphas_cumprod_t = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    @abstractmethod
    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single sampling step - must be implemented by subclasses
        
        Args:
            model: Denoising model
            x_t: Current noisy image
            t: Current timestep
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            
        Returns:
            x_t_minus_1: Denoised image at t-1
            x_0_pred: Optional prediction of clean image
        """
        pass
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        start_timestep: Optional[int] = None,
        return_intermediates: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Full sampling loop
        
        Args:
            model: Denoising model
            shape: Shape of image to generate
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled low-resolution image
            start_timestep: Starting timestep (default: num_timesteps-1)
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            Dictionary containing:
                - 'sample': Final generated image
                - 'intermediates': Optional list of intermediate samples
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Initialize with noise
        img = torch.randn(shape, device=device)
        
        # Setup timesteps for sampling
        timesteps = self.get_sampling_timesteps(start_timestep)
        
        # Progress bar
        iterator = tqdm(timesteps, desc=f'{self.__class__.__name__} sampling') if self.use_tqdm else timesteps
        
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
                intermediates.append({
                    't': t,
                    'sample': img.cpu(),
                    'pred_x0': x_0_pred.cpu() if x_0_pred is not None else None
                })
        
        results = {'sample': img}
        if return_intermediates:
            results['intermediates'] = intermediates
            
        return results
    
    def get_sampling_timesteps(self, start_timestep: Optional[int] = None) -> list:
        """Get timesteps for sampling - can be overridden for different sampling strategies"""
        if start_timestep is None:
            start_timestep = self.num_timesteps - 1
        return list(reversed(range(0, start_timestep + 1)))
    
    def img2res(self, x: torch.Tensor, img_lr_up: torch.Tensor, res_rescale: float = 2.0) -> torch.Tensor:
        """Convert image to residual for residual diffusion"""
        return (x - img_lr_up) * res_rescale
    
    def res2img(self, res: torch.Tensor, img_lr_up: torch.Tensor, res_rescale: float = 2.0) -> torch.Tensor:
        """Convert residual back to image"""
        img = res / res_rescale + img_lr_up
        if self.clip_denoised:
            img = torch.clamp(img, -1, 1)
        return img
