"""
Unified Frequency-Guided Sampling Framework for SRDiff
Consolidated implementation merging frequency_sampler.py and frequency_guided_wrappers.py
Provides both standalone frequency-aware sampler and wrapper capabilities for any base sampler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from .base_sampler import BaseSampler


class FrequencyGuidanceCore:
    """
    Core frequency guidance operations shared across all frequency-guided samplers.
    Consolidated from _FrequencyGuidanceOps and FrequencyAwareSampler implementations.
    """

    def __init__(
        self,
        # Data consistency parameters
        lambda_dc_max: float = 0.0,
        rho: float = 2.0,
        # Frequency guidance parameters
        lambda_freq_max: float = 0.0,
        kappa: float = 1.5,
        gamma: float = 5.5,
        delta: float = 1e-6,
        epsilon: float = 1e-8,
        # Sampling parameters
        eta_t: float = 0.1,
        # Frequency emphasis function
        freq_emphasis: str = 'exponential',
        freq_threshold: float = 0.25,
        **kwargs
    ):
        """
        Unified frequency guidance core with optimized default parameters.

        Args:
            lambda_dc_max: Maximum data consistency weight (early steps)
            rho: Exponent for DC schedule decay
            lambda_freq_max: Maximum frequency guidance weight (late steps)
            kappa: Exponent for freq schedule growth
            gamma: Frequency weight amplification factor
            delta: Numerical stability parameter
            epsilon: Power spectrum normalization parameter
            eta_t: Gradient descent step size
            freq_emphasis: Type of frequency emphasis ('exponential', 'quantile', 'linear')
            freq_threshold: Threshold for frequency emphasis
        """
        self.lambda_dc_max = lambda_dc_max
        self.rho = rho
        self.lambda_freq_max = lambda_freq_max
        self.kappa = kappa
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.eta_t = eta_t
        self.freq_emphasis = freq_emphasis
        self.freq_threshold = freq_threshold

        # Frequency weight cache for efficiency
        self._freq_weight_cache: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """Reset frequency weight cache for new sampling run."""
        self._freq_weight_cache = None

    def compute_schedule_weights(self, current_t: int, total_T: int) -> Tuple[float, float, float]:
        """
        Compute time-dependent schedule weights.

        Args:
            current_t: Current timestep (0 to T-1, where T-1 is full noise)
            total_T: Total number of timesteps

        Returns:
            lambda_dc: Data consistency weight
            lambda_freq: Frequency guidance weight
            alpha_t: Target amplitude scaling factor
        """
        # Normalize time to [0,1] where 1=noisiest, 0=clean
        t_norm = float(current_t) / float(total_T)

        # Optimized schedules:
        # - Data consistency grows towards late denoising (low noise)
        # - Frequency guidance stronger in early denoising (high noise)
        lambda_dc = self.lambda_dc_max * ((1.0 - t_norm) ** self.rho)
        lambda_freq = self.lambda_freq_max * ((1.0 - t_norm) ** self.kappa)

        # Adaptive alpha_t with sharp sigmoid for aggressive high-freq restoration
        alpha_t = torch.sigmoid(torch.tensor(10.0 * (0.3 - t_norm))).item()
        alpha_t = max(0.0, min(1.3, alpha_t + 0.2 * (1.0 - t_norm)))

        return lambda_dc, lambda_freq, alpha_t

    def compute_frequency_weight(self, y_lr: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Compute frequency-dependent weight W(ω) based on LR image spectrum.

        Args:
            y_lr: Low-resolution image [B, C, H, W]
            target_size: Target size (H, W) for upsampling. If None, uses input size.

        Returns:
            W: Frequency weight mask [B, C, H, W] in frequency domain
        """
        batch_size, channels, height, width = y_lr.shape

        # Compute 2D FFT and power spectrum of LR image
        Y = fft.fft2(y_lr, dim=(-2, -1))
        S_y = torch.abs(Y) ** 2

        # Normalize by total energy for stability
        S_y_norm = S_y / (S_y.sum(dim=(-2, -1), keepdim=True) + self.epsilon)

        # Apply frequency emphasis function φ
        if self.freq_emphasis == 'exponential':
            phi = torch.exp(S_y_norm / self.freq_threshold) - 1.0
        elif self.freq_emphasis == 'quantile':
            threshold = torch.quantile(S_y_norm.flatten(-2), self.freq_threshold, dim=-1, keepdim=True)
            threshold = threshold.unsqueeze(-1)
            phi = (S_y_norm > threshold).float() * S_y_norm
        else:  # linear
            phi = S_y_norm

        # Compute base frequency weight
        W = 1.0 + self.gamma * phi

        # Add multi-tier radial frequency mask for progressive emphasis
        cy, cx = height // 2, width // 2
        h_range = torch.arange(height, device=y_lr.device, dtype=torch.float32)
        w_range = torch.arange(width, device=y_lr.device, dtype=torch.float32)
        y_coords, x_coords = torch.meshgrid(h_range, w_range, indexing='ij')

        # Shift coordinates for FFT (zero frequency at corners)
        y_coords = torch.fft.ifftshift(y_coords - cy)
        x_coords = torch.fft.ifftshift(x_coords - cx)
        radius = torch.sqrt(y_coords**2 + x_coords**2) / max(height, width)

        # Enhanced radial frequency emphasis for sharpness
        radial_weight = torch.ones_like(radius)
        high_freq_mask = torch.sigmoid(8 * (radius - 0.3))  # Smooth transition at 0.3
        radial_weight += high_freq_mask * 0.7  # Strong boost for sharp details

        # Combine with radial mask and normalize
        W = W * radial_weight
        W = W / (W.mean(dim=(-2, -1), keepdim=True) + 1e-8)

        # Upsample to target size if provided
        if target_size is not None and (target_size[0] != height or target_size[1] != width):
            W = F.interpolate(W, size=target_size, mode='bicubic', align_corners=False)

        return W

    @staticmethod
    def bicubic_downsample(x: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
        """Bicubic downsampling operator D."""
        return F.interpolate(x, scale_factor=1.0/scale_factor, mode='bicubic', align_corners=False)

    @staticmethod
    def bicubic_upsample(y: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
        """Bicubic upsampling operator D^T (transpose/adjoint)."""
        return F.interpolate(y, scale_factor=scale_factor, mode='bicubic', align_corners=False)

    def compute_data_consistency_grad(
        self,
        x: torch.Tensor,
        y_lr: torch.Tensor,
        scale_factor: int = 4
    ) -> torch.Tensor:
        """
        Compute gradient of data consistency term: ∇L_dc = 2 * D^T(D(x) - y)

        Args:
            x: Current estimate [B, C, H, W]
            y_lr: Low-resolution observation [B, C, h, w]
            scale_factor: SR scale factor

        Returns:
            Gradient [B, C, H, W]
        """
        # Downsample current estimate and compute residual
        x_down = self.bicubic_downsample(x, scale_factor)
        residual = x_down - y_lr

        # Upsample residual to HR space (transpose operation)
        grad = 2.0 * self.bicubic_upsample(residual, scale_factor)
        return grad

    def compute_frequency_guidance_grad(
        self,
        x: torch.Tensor,
        y_up: torch.Tensor,
        alpha_t: float,
        W: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of frequency guidance term using Wirtinger calculus.
        ∇L_freq = 2 * Re[F^{-1}(W²(ω) * (|X| - α_t|Y_up|) * X/(|X|+δ))]

        Args:
            x: Current estimate [B, C, H, W]
            y_up: Upsampled LR image [B, C, H, W]
            alpha_t: Target amplitude scaling
            W: Frequency weight [B, C, H, W]

        Returns:
            Gradient [B, C, H, W]
        """
        # Compute FFT
        X = fft.fft2(x, dim=(-2, -1))
        Y_up = fft.fft2(y_up, dim=(-2, -1))

        # Compute magnitudes and target amplitude
        X_mag = torch.abs(X)
        Y_up_mag = torch.abs(Y_up)
        A = alpha_t * Y_up_mag

        # Amplitude error and normalized X
        amp_error = X_mag - A
        X_normalized = X / (X_mag + self.delta)

        # Frequency domain gradient
        G = 2.0 * (W ** 2) * amp_error * X_normalized

        # Convert to spatial domain (real part only)
        grad = torch.real(fft.ifft2(G, dim=(-2, -1)))
        return grad

    def compute_guidance_grad_residual(
        self,
        r: torch.Tensor,
        img_lr_up: torch.Tensor,
        img_lr: torch.Tensor,
        res_rescale: float,
        scale_factor: int,
        alpha_t: float,
        W: torch.Tensor,
        lambda_dc: float,
        lambda_freq: float,
    ) -> torch.Tensor:
        """Compute unified guidance gradient directly in residual space."""
        if abs(lambda_dc) < 1e-8 and abs(lambda_freq) < 1e-8:
            return torch.zeros_like(r)

        # Convert residual to image for gradient calculation
        x = r / res_rescale + img_lr_up
        grad_total = torch.zeros_like(x)

        # Add data consistency gradient
        if abs(lambda_dc) > 1e-8:
            grad_dc = self.compute_data_consistency_grad(x, img_lr, scale_factor)
            grad_total += lambda_dc * grad_dc

        # Add frequency guidance gradient
        if abs(lambda_freq) > 1e-8:
            grad_freq = self.compute_frequency_guidance_grad(x, img_lr_up, alpha_t, W)
            grad_total += lambda_freq * grad_freq

        # Convert gradient back to residual space using chain rule
        return grad_total / res_rescale


class FrequencyGuidedSampler(BaseSampler):
    """
    Unified frequency-guided sampler that can wrap any base sampling algorithm.
    Supports DDPM, DDIM, and DPM-Solver++ as base samplers with frequency guidance.
    """

    def __init__(
        self,
        base_sampler: str = 'ddim',  # 'ddpm', 'ddim', 'dpmpp'
        num_inference_steps: int = 25,
        # Base sampler specific parameters
        ddim_eta: float = 0.0,  # DDIM stochasticity (0=deterministic)
        solver_type: str = 'midpoint',  # DPM-Solver++ type
        # Frequency guidance parameters (optimized defaults)
        lambda_dc_max: float = 0.0,
        rho: float = 2.0,
        lambda_freq_max: float = 0.0,
        kappa: float = 1.5,
        gamma: float = 5.5,
        delta: float = 1e-6,
        epsilon: float = 1e-8,
        eta_t: float = 0.1,
        freq_emphasis: str = 'exponential',
        freq_threshold: float = 0.25,
        **kwargs
    ):
        """
        Unified frequency-guided sampler.

        Args:
            base_sampler: Base sampling algorithm ('ddpm', 'ddim', 'dpmpp')
            num_inference_steps: Number of denoising steps
            ddim_eta: DDIM stochasticity parameter
            solver_type: DPM-Solver++ solver type
            Other args: Frequency guidance parameters (see FrequencyGuidanceCore)
        """
        super().__init__(**kwargs)
        self.base_sampler = base_sampler
        self.num_inference_steps = num_inference_steps
        self.ddim_eta = ddim_eta
        self.solver_type = solver_type

        # Set sampler name
        base_names = {'ddpm': 'DDPM', 'ddim': 'DDIM', 'dpmpp': 'DPM++'}
        self.name = f"Freq-Guided {base_names.get(base_sampler, 'Unknown')}"

        # Initialize frequency guidance core
        self.x0_clip_range = 1.0
        self.guidance = FrequencyGuidanceCore(
            lambda_dc_max=lambda_dc_max,
            rho=rho,
            lambda_freq_max=lambda_freq_max,
            kappa=kappa,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            eta_t=eta_t,
            freq_emphasis=freq_emphasis,
            freq_threshold=freq_threshold
        )

        # Setup base sampler schedules
        if base_sampler == 'ddim':
            self.setup_ddim_schedule()
            # Preserve residual dynamic range; clamp only after full inversion
            self.clip_denoised = False
            self.x0_clip_range = 1.3
        elif base_sampler == 'dpmpp':
            self.setup_dpm_schedule()

    def setup_ddim_schedule(self):
        """Setup DDIM sampling schedule."""
        steps = max(1, int(self.num_inference_steps))
        timesteps = np.linspace(0, self.num_timesteps - 1, steps, dtype=np.int64)
        timesteps = np.flip(timesteps).copy()

        self.ddim_timesteps = timesteps
        self.ddim_timesteps_prev = np.append(timesteps[1:], 0)
        assert int(self.ddim_timesteps[-1]) == 0, "DDIM schedule must include t=0"

        # Precompute alpha values
        self.ddim_alpha = self.alphas_cumprod[timesteps]
        self.ddim_alpha_prev = torch.cat([
            self.alphas_cumprod[timesteps[1:]],
            torch.tensor([1.0], device=self.device)
        ])

        # Precompute sigma for variance with numeric guards
        self.ddim_sigma = (
            self.ddim_eta
            * torch.sqrt((1.0 - self.ddim_alpha_prev) / (1.0 - self.ddim_alpha + 1e-12))
            * torch.sqrt(torch.clamp(1.0 - self.ddim_alpha / (self.ddim_alpha_prev + 1e-12), min=0.0))
        )

    def setup_dpm_schedule(self) -> None:
        """Setup DPM-Solver++ schedule."""
        t = torch.linspace(0, self.num_timesteps - 1, self.num_inference_steps, device=self.device).flip(0)
        self.dpm_timesteps = t.long()
        assert int(self.dpm_timesteps[-1].item()) == 0, "DPM++ schedule must include t=0"

    def _ddpm_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard DDPM sampling step."""
        batch_size = x_t.shape[0]
        noise_pred = model(x_t, t, cond, img_lr_up)
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)

        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Posterior parameters
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # Sample
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise * temperature

        return x_prev, x_0_pred

    def _ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_idx: int,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDIM sampling step."""
        batch_size = x_t.shape[0]
        device = x_t.device

        # Get actual timestep value
        timestep = torch.full((batch_size,), self.ddim_timesteps[t_idx], device=device, dtype=torch.long)

        # Get noise prediction from model
        noise_pred = model(x_t, timestep, cond, img_lr_up)

        # Get alpha values
        alpha_t = self.ddim_alpha[t_idx]
        alpha_prev = self.ddim_alpha_prev[t_idx]
        sigma_t = self.ddim_sigma[t_idx]

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * noise_pred

        # Random noise (guard scalar tensor comparison)
        noise = torch.randn_like(x_t) if float(sigma_t.item()) > 0 else 0

        # Compute x_{t-1}
        x_t_minus_1 = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma_t * noise

        return x_t_minus_1, x_0_pred

    def _dpmpp_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        step_idx: int,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DPM-Solver++ midpoint step."""
        batch_size = x_t.shape[0]
        device = x_t.device

        t = self.dpm_timesteps[step_idx]
        t_prev_idx = min(step_idx + 1, len(self.dpm_timesteps) - 1)
        t_prev = self.dpm_timesteps[t_prev_idx]
        t_value = int(t.item())
        s_value = int(t_prev.item())
        if t_value == s_value:
            mid_value = t_value
        else:
            mid_value = int(round((t_value + s_value) / 2.0))
        mid_value = max(min(mid_value, t_value), s_value)

        # First prediction
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred_t = model(x_t, t_batch, cond, img_lr_up)

        # Standard parameterization
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        x_0_pred_t = (x_t - sigma_t * noise_pred_t) / sqrt_alpha_bar_t

        if self.clip_denoised:
            x_0_pred_t = torch.clamp(x_0_pred_t, -1.0, 1.0)
        else:
            x_0_pred_t = torch.clamp(x_0_pred_t, -self.x0_clip_range, self.x0_clip_range)

        # Midpoint prediction
        alpha_bar_mid = self.alphas_cumprod[mid_value]
        sqrt_alpha_bar_mid = torch.sqrt(alpha_bar_mid)
        sigma_mid = torch.sqrt(1 - alpha_bar_mid)
        x_mid = sqrt_alpha_bar_mid * x_0_pred_t + sigma_mid * noise_pred_t

        t_mid_batch = torch.full((batch_size,), mid_value, device=device, dtype=torch.long)
        noise_pred_mid = model(x_mid, t_mid_batch, cond, img_lr_up)

        # Final update
        alpha_bar_s = self.alphas_cumprod[s_value]
        sqrt_alpha_bar_s = torch.sqrt(alpha_bar_s)
        sigma_s = torch.sqrt(1 - alpha_bar_s)
        x_0_pred = (x_mid - sigma_mid * noise_pred_mid) / sqrt_alpha_bar_mid

        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        x_prev = sqrt_alpha_bar_s * x_0_pred + sigma_s * noise_pred_mid
        return x_prev, x_0_pred

    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        img_lr: torch.Tensor,
        t_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Unified frequency-guided sampling step.

        Args:
            model: Denoising model
            x_t: Current noisy state (residual)
            t: Current timestep
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled LR image
            img_lr: Original LR image
            t_idx: Timestep index for DDIM
            step_idx: Step index for DPM-Solver++

        Returns:
            x_guided: Guided next state
            x_0_pred: Predicted clean image
        """
        # Step 1: Base sampler update
        if self.base_sampler == 'ddim' and t_idx is not None:
            x_base, x_0_pred = self._ddim_step(model, x_t, t_idx, cond, img_lr_up)
        elif self.base_sampler == 'dpmpp' and step_idx is not None:
            x_base, x_0_pred = self._dpmpp_step(model, x_t, step_idx, cond, img_lr_up)
        else:  # DDPM
            x_base, x_0_pred = self._ddpm_step(model, x_t, t, cond, img_lr_up)

        # Step 2: Compute schedule weights
        if self.base_sampler == 'ddim' and t_idx is not None:
            current_t = int(self.ddim_timesteps[t_idx])
        elif self.base_sampler == 'dpmpp' and step_idx is not None:
            current_t = int(self.dpm_timesteps[step_idx].item())
        else:
            current_t = int(t[0].item())

        lambda_dc, lambda_freq, alpha_t = self.guidance.compute_schedule_weights(current_t, self.num_timesteps)

        # Step 3: Compute frequency weight (cached)
        if self.guidance._freq_weight_cache is None:
            target_shape = (img_lr_up.shape[2], img_lr_up.shape[3])
            self.guidance._freq_weight_cache = self.guidance.compute_frequency_weight(img_lr, target_size=target_shape)
        W = self.guidance._freq_weight_cache

        # Step 4: Compute guidance gradient
        res_rescale = float(kwargs.get('res_rescale', 2.0))
        scale_factor = int(kwargs.get('scale_factor', 4))

        grad_residual = self.guidance.compute_guidance_grad_residual(
            x_base, img_lr_up, img_lr, res_rescale, scale_factor, alpha_t, W, lambda_dc, lambda_freq
        )

        # Step 5: Apply guidance with clipping
        delta = self.guidance.eta_t * grad_residual
        max_step = float(kwargs.get('max_guidance_step', 0.1))
        mean_abs = delta.abs().mean(dim=(1, 2, 3), keepdim=True)
        scale = torch.clamp(mean_abs / (max_step + 1e-8), min=1.0)
        delta = delta / scale

        # Step 6: Apply guidance
        x_guided = x_base - delta

        return x_guided, x_0_pred

    def get_iterator(self, timesteps, desc=None):
        """Get iterator with optional progress bar."""
        if self.use_tqdm:
            from tqdm import tqdm
            return tqdm(timesteps, desc=desc or f'{self.name} sampling')
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        img_lr_up: torch.Tensor,
        img_lr: torch.Tensor,
        return_intermediates: bool = False,
        use_residual: bool = True,
        res_rescale: float = 2.0,
        **kwargs
    ):
        """
        Full unified frequency-guided sampling loop.

        Args:
            model: Denoising model
            shape: Shape of image to generate
            cond: Conditioning (RRDB features)
            img_lr_up: Upsampled LR image
            img_lr: Original LR image
            return_intermediates: Whether to return intermediate samples
            use_residual: Whether to use residual diffusion
            res_rescale: Residual rescaling factor

        Returns:
            Dictionary with results
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Reset frequency weight cache
        self.guidance.reset_cache()

        # Initialize with noise
        img = torch.randn(shape, device=device)

        # Get timesteps based on base sampler
        if self.base_sampler == 'ddim':
            timesteps = list(range(len(self.ddim_timesteps)))
            desc = f'{self.name} sampling (DDIM {self.num_inference_steps} steps)'
        elif self.base_sampler == 'dpmpp':
            timesteps = list(range(len(self.dpm_timesteps)))
            desc = f'{self.name} sampling (DPM++ {self.num_inference_steps} steps)'
        else:  # DDPM
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
            desc = f'{self.name} sampling (DDPM {self.num_timesteps} steps)'

        # Progress bar
        iterator = self.get_iterator(timesteps, desc)
        intermediates = []

        for idx in iterator:
            if self.base_sampler == 'ddim':
                t_batch = torch.full((batch_size,), self.ddim_timesteps[idx], device=device, dtype=torch.long)
                img, x_0_pred = self.sample_step(
                    model=model, x_t=img, t=t_batch, cond=cond,
                    img_lr_up=img_lr_up, img_lr=img_lr, t_idx=idx, **kwargs
                )
            elif self.base_sampler == 'dpmpp':
                t_batch = torch.zeros((batch_size,), device=device, dtype=torch.long)  # Placeholder
                img, x_0_pred = self.sample_step(
                    model=model, x_t=img, t=t_batch, cond=cond,
                    img_lr_up=img_lr_up, img_lr=img_lr, step_idx=idx, **kwargs
                )
            else:  # DDPM
                t_batch = torch.full((batch_size,), idx, device=device, dtype=torch.long)
                img, x_0_pred = self.sample_step(
                    model=model, x_t=img, t=t_batch, cond=cond,
                    img_lr_up=img_lr_up, img_lr=img_lr, **kwargs
                )

            # Save intermediates
            if return_intermediates and idx % 5 == 0:
                if use_residual:
                    img_vis = self.res2img(img, img_lr_up, res_rescale)
                    x_0_vis = self.res2img(x_0_pred, img_lr_up, res_rescale) if x_0_pred is not None else None
                else:
                    img_vis = img
                    x_0_vis = x_0_pred

                intermediates.append({
                    't': t_batch[0].item() if self.base_sampler != 'dpmpp' else idx,
                    'sample': img_vis.cpu(),
                    'pred_x0': x_0_vis.cpu() if x_0_vis is not None else None
                })

        # Convert final residual to image
        if use_residual:
            img = self.res2img(img, img_lr_up, res_rescale)
        img = torch.clamp(img, -1.0, 1.0)

        results = {
            'sample': img,
            'method': 'frequency_guided',
            'base': self.base_sampler,
            'num_steps': self.num_inference_steps if self.base_sampler != 'ddpm' else self.num_timesteps
        }

        if return_intermediates:
            results['intermediates'] = intermediates

        return results


# Legacy compatibility aliases
FrequencyAwareSampler = FrequencyGuidedSampler  # For backward compatibility

# Convenient factory functions for specific base samplers
def create_frequency_guided_ddpm(**kwargs):
    """Create frequency-guided DDPM sampler."""
    return FrequencyGuidedSampler(base_sampler='ddpm', **kwargs)

def create_frequency_guided_ddim(**kwargs):
    """Create frequency-guided DDIM sampler."""
    return FrequencyGuidedSampler(base_sampler='ddim', **kwargs)

def create_frequency_guided_dpmpp(**kwargs):
    """Create frequency-guided DPM-Solver++ sampler."""
    return FrequencyGuidedSampler(base_sampler='dpmpp', **kwargs)
