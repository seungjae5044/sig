"""
Evaluator for SRDiff Multi-Sampler System
Handles batch processing and comprehensive evaluation across multiple test datasets
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.diffsr_modules import Unet, RRDBNet
from models.diffusion import GaussianDiffusion
from utils.hparams import hparams, set_hparams

from metrics import MetricsCalculator
from samplers.base_sampler import BaseSampler
from samplers.ddpm_sampler import DDPMSampler
from samplers.ddim_sampler import DDIMSampler
from samplers.dpmpp_sampler import DPMPPSampler
from samplers.unified_frequency_guidance import (
    FrequencyGuidedSampler,
    FrequencyAwareSampler,
    create_frequency_guided_ddpm,
    create_frequency_guided_ddim,
    create_frequency_guided_dpmpp,
)


class SuperResolutionDataset(Dataset):
    """Dataset for loading HR/LR image pairs from test datasets"""

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        scale_factor: int = 4,
        max_images: Optional[int] = None,
    ):
        """
        Args:
            dataset_root: Root directory containing test datasets
            dataset_name: Name of dataset (BSD100, Set14, Set5, Urban100)
            scale_factor: Upsampling scale factor (2, 3, 4)
            max_images: Maximum number of images to load (for testing)
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.scale_factor = scale_factor

        # Construct paths
        self.dataset_dir = (
            self.dataset_root / dataset_name / f"image_SRF_{scale_factor}"
        )
        self.hr_dir = self.dataset_dir / "HR"
        self.lr_dir = self.dataset_dir / "LR"

        # Check if directories exist
        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise ValueError(f"Dataset directories not found: {self.dataset_dir}")

        # Get image pairs
        self.image_pairs = self._get_image_pairs()

        if max_images is not None:
            self.image_pairs = self.image_pairs[:max_images]

        print(
            f"Loaded {len(self.image_pairs)} image pairs from {dataset_name} (scale {scale_factor}x)"
        )

    def _get_image_pairs(self) -> List[Dict[str, str]]:
        """Get list of HR/LR image pairs"""
        pairs = []

        # Get all HR images
        hr_files = sorted(
            list(self.hr_dir.glob("*.png"))
            + list(self.hr_dir.glob("*.jpg"))
            + list(self.hr_dir.glob("*.jpeg"))
            + list(self.hr_dir.glob("*.bmp"))
        )

        for hr_file in hr_files:
            lr_file = self._find_lr_pair(hr_file)

            if lr_file is not None:
                pairs.append(
                    {
                        "hr_path": str(hr_file),
                        "lr_path": str(lr_file),
                        "image_name": self._normalize_image_name(hr_file),
                    }
                )

        return pairs

    def _find_lr_pair(self, hr_file: Path) -> Optional[Path]:
        candidates = []
        if "_HR" in hr_file.name:
            candidates.append(self.lr_dir / hr_file.name.replace("_HR", "_LR"))
        candidates.append(self.lr_dir / hr_file.name)
        candidates.append(
            self.lr_dir / f"{hr_file.stem}x{self.scale_factor}{hr_file.suffix}"
        )
        candidates.append(
            self.lr_dir / f"{hr_file.stem}_x{self.scale_factor}{hr_file.suffix}"
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _normalize_image_name(self, hr_file: Path) -> str:
        stem = hr_file.stem
        for suffix in (f"_SRF_{self.scale_factor}_HR", "_HR"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _ensure_divisible_size(
        self, tensor: torch.Tensor, divisor: int = 32
    ) -> torch.Tensor:
        """Ensure tensor size is divisible by divisor by padding"""
        _, h, w = tensor.shape

        # Calculate new dimensions
        new_h = ((h + divisor - 1) // divisor) * divisor
        new_w = ((w + divisor - 1) // divisor) * divisor

        # Pad if necessary
        if h != new_h or w != new_w:
            pad_h = new_h - h
            pad_w = new_w - w
            # Pad: (left, right, top, bottom)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        return tensor

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.image_pairs[idx]

        # Load images
        hr_img = Image.open(pair["hr_path"]).convert("RGB")
        lr_img = Image.open(pair["lr_path"]).convert("RGB")

        # Convert to tensors
        hr_tensor = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0

        # Ensure dimensions are divisible by 32 for UNet compatibility
        hr_tensor = self._ensure_divisible_size(hr_tensor, divisor=32)
        lr_tensor = self._ensure_divisible_size(
            lr_tensor, divisor=8
        )  # LR needs smaller divisor

        # Create upsampled LR for conditioning
        lr_up = torch.nn.functional.interpolate(
            lr_tensor.unsqueeze(0),
            size=hr_tensor.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Normalize to [-1, 1] for diffusion models
        hr_tensor = hr_tensor * 2.0 - 1.0
        lr_tensor = lr_tensor * 2.0 - 1.0
        lr_up = lr_up * 2.0 - 1.0

        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "lr_up": lr_up,
            "image_name": pair["image_name"],
            "dataset": self.dataset_name,
            "scale_factor": self.scale_factor,
        }


class SRDiffEvaluator:
    """Main evaluator for SRDiff multi-sampler system"""

    def __init__(
        self,
        config_path: str,
        model_ckpt_path: str,
        rrdb_ckpt_path: Optional[str] = None,
        device: str = "cuda",
        output_dir: str = "./results",
    ):
        """
        Args:
            config_path: Path to SRDiff config file
            model_ckpt_path: Path to pretrained SRDiff checkpoint
            rrdb_ckpt_path: Path to pretrained RRDB checkpoint (optional, RRDB will be loaded from SRDiff checkpoint)
            device: Device for computation
            output_dir: Output directory for results
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        set_hparams(config_path)

        # Initialize model
        self.model = self._load_model(model_ckpt_path, rrdb_ckpt_path)

        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(device=device)

        # Initialize samplers
        self.samplers = self._initialize_samplers()

        print(f"SRDiff Evaluator initialized with {len(self.samplers)} samplers")

    def _load_model(
        self, model_ckpt_path: str, rrdb_ckpt_path: Optional[str] = None
    ) -> GaussianDiffusion:
        """Load pretrained SRDiff model"""
        # Analyze main checkpoint to get correct RRDB parameters
        rrdb_nf, rrdb_nb, rrdb_gc = 32, 8, 16  # defaults
        if os.path.exists(model_ckpt_path):
            try:
                main_ckpt = torch.load(model_ckpt_path, map_location="cpu")
                if "state_dict" in main_ckpt and "model" in main_ckpt["state_dict"]:
                    model_state = main_ckpt["state_dict"]["model"]
                    rrdb_keys = [k for k in model_state.keys() if k.startswith("rrdb.")]

                    # Get rrdb_num_feat from conv_first
                    if "rrdb.conv_first.weight" in model_state:
                        rrdb_nf = model_state["rrdb.conv_first.weight"].shape[0]

                    # Count RRDB blocks
                    max_block = -1
                    for key in rrdb_keys:
                        if "RRDB_trunk." in key:
                            try:
                                block_num = int(key.split(".")[2])  # rrdb.RRDB_trunk.X.
                                max_block = max(max_block, block_num)
                            except:
                                pass
                    if max_block >= 0:
                        rrdb_nb = max_block + 1

                    # Get gc from RDB conv
                    for key in rrdb_keys:
                        if "RDB1.conv1.weight" in key:
                            rrdb_gc = model_state[key].shape[0]
                            break

                    print(
                        f"Detected RRDB parameters from main checkpoint: nf={rrdb_nf}, nb={rrdb_nb}, gc={rrdb_gc}"
                    )
            except Exception as e:
                print(f"Warning: Could not analyze main checkpoint: {e}")
                print("Using default RRDB parameters")

        # Now initialize UNet with correct conditioning dimension
        hidden_size = hparams.get("hidden_size", 64)
        dim_mults = hparams.get("unet_dim_mults", "1|2|2|4")
        dim_mults = [int(x) for x in dim_mults.split("|")]

        denoise_fn = Unet(
            hidden_size,
            out_dim=3,
            cond_dim=rrdb_nf,  # Use detected RRDB feature size
            dim_mults=dim_mults,
        )

        # Initialize RRDB with detected parameters
        rrdb = RRDBNet(in_nc=3, out_nc=3, nf=rrdb_nf, nb=rrdb_nb, gc=rrdb_gc)

        # Note: RRDB weights will be loaded from main checkpoint
        if rrdb_ckpt_path:
            print(
                f"RRDB checkpoint path provided but will be ignored: {rrdb_ckpt_path}"
            )
        print(f"RRDB will be loaded from main SRDiff checkpoint")

        # Create diffusion model (use pretrained model's original timesteps for compatibility)
        model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=hparams.get(
                "timesteps", 100
            ),  # Use original pretrained timesteps
            loss_type=hparams.get("loss_type", "l1"),
        )

        # Load main model checkpoint with proper nested structure handling
        if os.path.exists(model_ckpt_path):
            try:
                checkpoint = torch.load(model_ckpt_path, map_location="cpu")
                if "state_dict" in checkpoint and "model" in checkpoint["state_dict"]:
                    # Handle nested model state_dict structure
                    model_state = checkpoint["state_dict"]["model"]
                    model.load_state_dict(model_state)
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Successfully loaded SRDiff model checkpoint: {model_ckpt_path}")
            except Exception as e:
                print(f"Error loading SRDiff model checkpoint: {e}")
                print("Continuing with randomly initialized model...")
        else:
            print(f"Warning: Model checkpoint not found: {model_ckpt_path}")

        model.to(self.device)
        model.eval()

        return model

    def _initialize_samplers(self) -> Dict[str, BaseSampler]:
        """Initialize all available samplers"""
        samplers = {}
        # Prepare precomputed diffusion schedule from the loaded model to
        # ensure samplers use the exact same alphas/betas as training.
        sched = {
            "betas": self.model.betas,
            "alphas_cumprod": self.model.alphas_cumprod,
            "alphas_cumprod_prev": self.model.alphas_cumprod_prev,
            "sqrt_alphas_cumprod": self.model.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.model.sqrt_one_minus_alphas_cumprod,
            "sqrt_recip_alphas_cumprod": self.model.sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": self.model.sqrt_recipm1_alphas_cumprod,
            "posterior_variance": self.model.posterior_variance,
            "posterior_log_variance_clipped": self.model.posterior_log_variance_clipped,
            "posterior_mean_coef1": self.model.posterior_mean_coef1,
            "posterior_mean_coef2": self.model.posterior_mean_coef2,
        }

        # DDPM (baseline)
        samplers["ddpm_100"] = DDPMSampler(
            num_timesteps=hparams.get("timesteps", 1000),
            device=self.device,
            use_tqdm=True,
            precomputed_schedule=sched,
        )

        # DDIM variants
        for steps in [5, 10, 20, 25, 50, 100]:
            samplers[f"ddim_{steps}"] = DDIMSampler(
                num_timesteps=hparams.get("timesteps", 1000),
                num_inference_steps=steps,
                eta=0.0,  # Deterministic
                device=self.device,
                use_tqdm=True,
                precomputed_schedule=sched,
            )

        # DPM-Solver++ variants
        for steps in [5, 10, 20, 25, 50, 100]:
            samplers[f"dpmpp_{steps}"] = DPMPPSampler(
                num_timesteps=hparams.get("timesteps", 1000),
                num_inference_steps=steps,
                solver_order=2,
                device=self.device,
                use_tqdm=True,
                precomputed_schedule=sched,
            )

        # Frequency-guided DDPM base
        samplers["freq_guided_ddpm"] = create_frequency_guided_ddpm(
            num_timesteps=hparams.get("timesteps", 1000),
            lambda_dc_max=0.01,
            lambda_freq_max=0.25,
            eta_t=0.01,
            gamma=0.2,
            device=self.device,
            use_tqdm=True,
            precomputed_schedule=sched,
        )

        # Frequency-guided DDIM base (midpoint)
        for steps in [5, 10, 20, 25, 50, 100]:
            samplers[f"freq_guided_ddim_{steps}"] = create_frequency_guided_ddim(
                num_timesteps=hparams.get("timesteps", 1000),
                num_inference_steps=steps,
                lambda_dc_max=0.01,
                lambda_freq_max=0.25,
                eta_t=0.005,
                gamma=0.2,
                device=self.device,
                use_tqdm=True,
                precomputed_schedule=sched,
            )

        # Frequency-guided DPM-Solver++ base (midpoint)
        for steps in [5, 10, 20, 25, 50, 100]:
            samplers[f"freq_guided_dpmpp_{steps}"] = create_frequency_guided_dpmpp(
                num_timesteps=hparams.get("timesteps", 1000),
                num_inference_steps=steps,
                solver_type="midpoint",
                lambda_dc_max=0.005,
                lambda_freq_max=0.2,
                eta_t=0.01,
                gamma=0.2,
                device=self.device,
                use_tqdm=True,
                precomputed_schedule=sched,
            )

        return samplers

    @torch.no_grad()
    def evaluate_single_image_with_sr(
        self,
        sample: Dict[str, Any],
        sampler_name: str,
        save_images: bool = True,
        # Patch processing options
        use_patches: bool = False,
        patch_size: int = 160,
        patch_overlap: int = 40,
        patch_batch_size: int = 4,
        auto_patch_threshold: int = 2048,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Evaluate single image with specified sampler and return SR tensor"""

        def _sync_cuda():
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        # Check if we should use patch processing
        hr_height, hr_width = sample["hr"].shape[1], sample["hr"].shape[2]
        should_use_patches = use_patches or (
            max(hr_height, hr_width) > auto_patch_threshold
        )

        if should_use_patches:
            result_dict, sr_eval = self.evaluate_single_image_patched(
                sample=sample,
                sampler_name=sampler_name,
                patch_size=patch_size,
                overlap=patch_overlap,
                patch_batch_size=patch_batch_size,
                save_images=save_images,
            )
            return result_dict, sr_eval

        # Original processing
        sampler = self.samplers[sampler_name]

        # Prepare inputs
        hr = sample["hr"].unsqueeze(0).to(self.device)
        lr = sample["lr"].unsqueeze(0).to(self.device)
        lr_up = sample["lr_up"].unsqueeze(0).to(self.device)

        # Get RRDB conditioning
        with torch.no_grad():
            if hparams.get("use_rrdb", True):
                rrdb_out, cond = self.model.rrdb(lr, True)
            else:
                rrdb_out = lr_up
                cond = lr

        # Sample with diffusion model
        inference_time = 0.0
        if isinstance(sampler, (FrequencyAwareSampler, FrequencyGuidedSampler)):
            # Frequency-guided samplers need LR image for gradient computation
            _sync_cuda()
            t0 = time.perf_counter()
            result = sampler.sample(
                model=self.model.denoise_fn,
                shape=hr.shape,
                cond=cond,
                img_lr_up=lr_up,
                img_lr=lr,
                use_residual=hparams.get("res", True),
                res_rescale=hparams.get("res_rescale", 2.0),
                scale_factor=sample["scale_factor"],
            )
            _sync_cuda()
            inference_time = time.perf_counter() - t0
            sr_img = result["sample"]
            extra_info = {
                "method": result.get("method", "frequency_guided"),
                "base": result.get("base", "ddim"),
                "num_steps": result.get("num_steps"),
            }
        else:
            # Other samplers
            _sync_cuda()
            t0 = time.perf_counter()
            result = sampler.sample(
                model=self.model.denoise_fn,
                shape=hr.shape,
                cond=cond,
                img_lr_up=lr_up,
                use_residual=hparams.get("res", True),
                res_rescale=hparams.get("res_rescale", 2.0),
            )
            _sync_cuda()
            inference_time = time.perf_counter() - t0
            sr_img = result["sample"]
            extra_info = {}

        # Convert back to [0, 1] range for metrics
        hr_eval = (hr + 1.0) / 2.0
        sr_eval = (sr_img + 1.0) / 2.0
        lr_eval = (lr + 1.0) / 2.0

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            img_hr=hr_eval.squeeze(0),
            img_sr=sr_eval.squeeze(0),
            img_lr=lr_eval.squeeze(0),
            scale_factor=sample["scale_factor"],
            calculate_fid=False,  # FID calculated separately for batches
        )

        # Save images if requested
        if save_images:
            self._save_sr_image(
                sr_eval.squeeze(0),
                sample["dataset"],
                sample["image_name"],
                sampler_name,
            )

        # Prepare result
        result_dict = {
            "Dataset": sample["dataset"],
            "Image": sample["image_name"],
            "Sampler": sampler_name,
            "Scale": sample["scale_factor"],
            "time": inference_time,
            **metrics,
            **extra_info,
        }

        return result_dict, sr_eval.squeeze(0)

    def _split_image_to_patches(
        self,
        img_tensor: torch.Tensor,
        patch_size: int,
        overlap: int = 40,
        scale_factor: int = 1,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """
        Split image tensor into overlapping patches following SRDiff's approach

        Args:
            img_tensor: Input tensor of shape [C, H, W]
            patch_size: Size of each patch (will be adjusted for scale_factor)
            overlap: Overlap between patches in pixels
            scale_factor: Scale factor for LR images (1 for HR, 4 for LR)

        Returns:
            patches: List of patch tensors
            positions: List of (x, y, x_end, y_end) positions for each patch
        """
        c, h, w = img_tensor.shape
        effective_patch_size = (
            patch_size // scale_factor if scale_factor > 1 else patch_size
        )
        effective_overlap = overlap // scale_factor if scale_factor > 1 else overlap

        # Calculate step size (patch_size - overlap) and guard against invalid configs
        step = effective_patch_size - effective_overlap
        if step <= 0:
            # Fallback: ensure forward progress
            step = max(1, effective_patch_size // 2)
            effective_overlap = effective_patch_size - step

        patches: List[torch.Tensor] = []
        positions: List[Tuple[int, int, int, int]] = []

        # Ensure image dimensions are compatible with scale_factor
        if scale_factor > 1:
            h = h - (h % scale_factor)
            w = w - (w % scale_factor)

        # Compute stable start grids and ensure coverage of image borders
        def compute_starts(limit: int) -> List[int]:
            if limit <= effective_patch_size:
                return [0]
            starts = list(range(0, limit - effective_patch_size + 1, step))
            last_start = limit - effective_patch_size
            if starts[-1] != last_start:
                starts.append(last_start)
            return starts

        ys = compute_starts(h)
        xs = compute_starts(w)

        for y in ys:
            for x in xs:
                y_end = min(y + effective_patch_size, h)
                x_end = min(x + effective_patch_size, w)

                # Extract and pad to consistent size if needed
                patch = img_tensor[:, y:y_end, x:x_end]
                if (
                    patch.shape[1] < effective_patch_size
                    or patch.shape[2] < effective_patch_size
                ):
                    pad_h = max(0, effective_patch_size - patch.shape[1])
                    pad_w = max(0, effective_patch_size - patch.shape[2])
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

                patches.append(patch)
                positions.append((x, y, x_end, y_end))

        return patches, positions

    def _merge_patches_with_overlap(
        self,
        patches: List[torch.Tensor],
        positions: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int, int],
        overlap: int = 40,
        scale_factor: int = 1,
    ) -> torch.Tensor:
        """
        Merge overlapping patches back into a single image with Gaussian blending

        Args:
            patches: List of patch tensors
            positions: List of (x, y, x_end, y_end) positions for each patch
            output_shape: Target output shape [C, H, W]
            overlap: Overlap between patches in pixels
            scale_factor: Scale factor for adjustment

        Returns:
            merged: Merged image tensor
        """
        c, h, w = output_shape
        effective_overlap = overlap // scale_factor if scale_factor > 1 else overlap

        # Initialize output tensor and weight map
        merged = torch.zeros(
            output_shape, dtype=patches[0].dtype, device=patches[0].device
        )
        weight_map = torch.zeros((h, w), dtype=torch.float32, device=patches[0].device)

        # Create weight mask for blending using distance transform
        def create_weight_mask(
            patch_h: int, patch_w: int, overlap: int
        ) -> torch.Tensor:
            """Create distance-based weight mask for smooth blending"""
            # Create distance from edge map
            y_coords = torch.arange(patch_h, dtype=torch.float32)
            x_coords = torch.arange(patch_w, dtype=torch.float32)

            # Distance from each edge
            dist_top = y_coords
            dist_bottom = patch_h - 1 - y_coords
            dist_left = x_coords
            dist_right = patch_w - 1 - x_coords

            # Minimum distance to any edge
            dist_y = torch.minimum(dist_top, dist_bottom)
            dist_x = torch.minimum(dist_left, dist_right)

            # Create 2D distance map (distance to nearest edge)
            dist_y_2d = dist_y.view(-1, 1).expand(patch_h, patch_w)
            dist_x_2d = dist_x.view(1, -1).expand(patch_h, patch_w)
            min_dist = torch.minimum(dist_y_2d, dist_x_2d)

            # Convert distance to weight (sigmoid-like function)
            if overlap > 0:
                weight = torch.clamp(min_dist / overlap, 0.0, 1.0)
                # Apply smooth transition
                weight = torch.where(
                    weight < 1.0,
                    0.5 * (1 + torch.cos(torch.pi * (1 - weight))),
                    torch.ones_like(weight),
                )
            else:
                weight = torch.ones((patch_h, patch_w), dtype=torch.float32)

            return weight

        # Process each patch
        for patch, (x, y, x_end, y_end) in zip(patches, positions):
            patch_h, patch_w = y_end - y, x_end - x

            # Get the actual patch content (remove padding if any)
            patch_content = patch[:, :patch_h, :patch_w]

            # Create weight mask for this patch
            weight_mask = create_weight_mask(patch_h, patch_w, effective_overlap)
            weight_mask = weight_mask.to(patches[0].device)

            # Add weighted patch to output
            merged[:, y:y_end, x:x_end] += patch_content * weight_mask.unsqueeze(0)
            weight_map[y:y_end, x:x_end] += weight_mask

        # Normalize by weight map to get final result
        weight_map = torch.clamp(weight_map, min=1e-8)  # Avoid division by zero
        merged = merged / weight_map.unsqueeze(0)

        return merged

    @torch.no_grad()
    def evaluate_single_image_patched(
        self,
        sample: Dict[str, Any],
        sampler_name: str,
        patch_size: int = 160,
        overlap: int = 40,
        patch_batch_size: int = 4,
        save_images: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate single image using patch-based processing"""

        def _sync_cuda():
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        sampler = self.samplers[sampler_name]
        scale_factor = sample["scale_factor"]

        # Get original tensors (without batch dimension)
        hr_orig = sample["hr"].to(self.device)
        lr_orig = sample["lr"].to(self.device)
        lr_up_orig = sample["lr_up"].to(self.device)

        print(
            f"Processing {sample['image_name']} with patches ({hr_orig.shape[1]}x{hr_orig.shape[2]} -> patches of {patch_size}x{patch_size})"
        )

        # Split images into patches
        hr_patches, hr_positions = self._split_image_to_patches(
            hr_orig, patch_size, overlap, scale_factor=1
        )
        lr_patches, lr_positions = self._split_image_to_patches(
            lr_orig, patch_size, overlap, scale_factor=scale_factor
        )
        lr_up_patches, lr_up_positions = self._split_image_to_patches(
            lr_up_orig, patch_size, overlap, scale_factor=1
        )

        print(f"Split into {len(hr_patches)} patches")

        # Process patches in batches
        sr_patches = []
        num_patches = len(hr_patches)
        inference_time = 0.0

        with tqdm(
            total=num_patches, desc=f"Processing patches ({sampler_name})"
        ) as pbar:
            for i in range(0, num_patches, patch_batch_size):
                batch_end = min(i + patch_batch_size, num_patches)
                batch_hr = torch.stack(hr_patches[i:batch_end]).to(self.device)
                batch_lr = torch.stack(lr_patches[i:batch_end]).to(self.device)
                batch_lr_up = torch.stack(lr_up_patches[i:batch_end]).to(self.device)

                # Get RRDB conditioning for batch
                with torch.no_grad():
                    if hparams.get("use_rrdb", True):
                        _, batch_cond = self.model.rrdb(batch_lr, True)
                    else:
                        batch_cond = batch_lr

                # Sample with diffusion model
                if isinstance(sampler, (FrequencyAwareSampler, FrequencyGuidedSampler)):
                    # Frequency-guided samplers need LR image for gradient computation
                    _sync_cuda()
                    t0 = time.perf_counter()
                    result = sampler.sample(
                        model=self.model.denoise_fn,
                        shape=batch_hr.shape,
                        cond=batch_cond,
                        img_lr_up=batch_lr_up,
                        img_lr=batch_lr,
                        use_residual=hparams.get("res", True),
                        res_rescale=hparams.get("res_rescale", 2.0),
                        scale_factor=scale_factor,
                    )
                    _sync_cuda()
                    inference_time += time.perf_counter() - t0
                    batch_sr = result["sample"]
                    extra_info = {
                        "method": result.get("method", "frequency_guided"),
                        "base": result.get("base", "ddim"),
                        "num_steps": result.get("num_steps"),
                    }
                else:
                    # Other samplers
                    _sync_cuda()
                    t0 = time.perf_counter()
                    result = sampler.sample(
                        model=self.model.denoise_fn,
                        shape=batch_hr.shape,
                        cond=batch_cond,
                        img_lr_up=batch_lr_up,
                        use_residual=hparams.get("res", True),
                        res_rescale=hparams.get("res_rescale", 2.0),
                    )
                    _sync_cuda()
                    inference_time += time.perf_counter() - t0
                    batch_sr = result["sample"]
                    extra_info = {}

                # Add patches to list
                for j in range(batch_sr.shape[0]):
                    sr_patches.append(batch_sr[j])

                pbar.update(batch_end - i)

        # Merge patches back into full image
        print(f"Merging {len(sr_patches)} patches...")
        sr_merged = self._merge_patches_with_overlap(
            sr_patches, hr_positions, hr_orig.shape, overlap, scale_factor=1
        )

        # Convert back to [0, 1] range for metrics
        hr_eval = (hr_orig + 1.0) / 2.0
        sr_eval = (sr_merged + 1.0) / 2.0
        lr_eval = (lr_orig + 1.0) / 2.0

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            img_hr=hr_eval,
            img_sr=sr_eval,
            img_lr=lr_eval,
            scale_factor=scale_factor,
            calculate_fid=False,
        )

        # Save images if requested
        if save_images:
            self._save_sr_image(
                sr_eval, sample["dataset"], sample["image_name"], sampler_name
            )

        # Prepare result
        result_dict = {
            "Dataset": sample["dataset"],
            "Image": sample["image_name"],
            "Sampler": sampler_name,
            "Scale": scale_factor,
            "Num_Patches": len(sr_patches),
            "Patch_Size": patch_size,
            "time": inference_time,
            **metrics,
            **extra_info,
        }

        return result_dict, sr_eval

    @torch.no_grad()
    def evaluate_single_image(
        self,
        sample: Dict[str, Any],
        sampler_name: str,
        save_images: bool = True,
        # Patch processing options
        use_patches: bool = False,
        patch_size: int = 160,
        overlap: int = 40,
        patch_batch_size: int = 4,
        auto_patch_threshold: int = 2048,
    ) -> Dict[str, Any]:
        """Evaluate single image with specified sampler"""

        def _sync_cuda():
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        # Check if we should use patch processing
        hr_height, hr_width = sample["hr"].shape[1], sample["hr"].shape[2]
        should_use_patches = use_patches or (
            max(hr_height, hr_width) > auto_patch_threshold
        )

        if should_use_patches:
            result_dict, _ = self.evaluate_single_image_patched(
                sample=sample,
                sampler_name=sampler_name,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                save_images=save_images,
            )
            return result_dict

        # Original single-image processing
        sampler = self.samplers[sampler_name]

        # Prepare inputs
        hr = sample["hr"].unsqueeze(0).to(self.device)
        lr = sample["lr"].unsqueeze(0).to(self.device)
        lr_up = sample["lr_up"].unsqueeze(0).to(self.device)

        # Get RRDB conditioning
        with torch.no_grad():
            if hparams.get("use_rrdb", True):
                rrdb_out, cond = self.model.rrdb(lr, True)
            else:
                rrdb_out = lr_up
                cond = lr

        # Sample with diffusion model
        inference_time = 0.0
        if isinstance(sampler, (FrequencyAwareSampler, FrequencyGuidedSampler)):
            # Frequency-guided samplers need LR image for gradient computation
            _sync_cuda()
            t0 = time.perf_counter()
            result = sampler.sample(
                model=self.model.denoise_fn,
                shape=hr.shape,
                cond=cond,
                img_lr_up=lr_up,
                img_lr=lr,
                use_residual=hparams.get("res", True),
                res_rescale=hparams.get("res_rescale", 2.0),
                scale_factor=sample["scale_factor"],
            )
            _sync_cuda()
            inference_time = time.perf_counter() - t0
            sr_img = result["sample"]
            extra_info = {
                "method": result.get("method", "frequency_guided"),
                "base": result.get("base", "ddim"),
                "num_steps": result.get("num_steps"),
            }
        else:
            # Other samplers
            _sync_cuda()
            t0 = time.perf_counter()
            result = sampler.sample(
                model=self.model.denoise_fn,
                shape=hr.shape,
                cond=cond,
                img_lr_up=lr_up,
                use_residual=hparams.get("res", True),
                res_rescale=hparams.get("res_rescale", 2.0),
            )
            _sync_cuda()
            inference_time = time.perf_counter() - t0
            sr_img = result["sample"]
            extra_info = {}

        # Convert back to [0, 1] range for metrics
        hr_eval = (hr + 1.0) / 2.0
        sr_eval = (sr_img + 1.0) / 2.0
        lr_eval = (lr + 1.0) / 2.0

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            img_hr=hr_eval.squeeze(0),
            img_sr=sr_eval.squeeze(0),
            img_lr=lr_eval.squeeze(0),
            scale_factor=sample["scale_factor"],
            calculate_fid=False,  # FID calculated separately for batches
        )

        # Save images if requested
        if save_images:
            self._save_sr_image(
                sr_eval.squeeze(0),
                sample["dataset"],
                sample["image_name"],
                sampler_name,
            )

        # Prepare result
        result_dict = {
            "Dataset": sample["dataset"],
            "Image": sample["image_name"],
            "Sampler": sampler_name,
            "Scale": sample["scale_factor"],
            "time": inference_time,
            **metrics,
            **extra_info,
        }

        return result_dict

    def _save_sr_image(
        self, sr_tensor: torch.Tensor, dataset: str, image_name: str, sampler_name: str
    ):
        """Save super-resolved image"""
        # Create output directory
        output_path = self.output_dir / "images" / dataset / sampler_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert tensor to PIL image
        sr_np = (sr_tensor.cpu().numpy() * 255).astype(np.uint8)
        sr_np = np.transpose(sr_np, (1, 2, 0))  # CHW -> HWC
        sr_pil = Image.fromarray(sr_np)

        # Save image
        filename = f"{image_name}_{sampler_name}.png"
        sr_pil.save(output_path / filename)

    def evaluate_dataset(
        self,
        dataset_root: str,
        dataset_name: str,
        scale_factor: int = 4,
        sampler_names: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        save_images: bool = True,
        calculate_fid: bool = True,
        batch_size: int = 1,
        # Patch processing options
        use_patches: bool = False,
        patch_size: int = 160,
        patch_overlap: int = 40,
        patch_batch_size: int = 4,
        auto_patch_threshold: int = 2048,
    ) -> List[Dict[str, Any]]:
        """Evaluate entire dataset with specified samplers"""
        if sampler_names is None:
            sampler_names = list(self.samplers.keys())

        # Load dataset
        dataset = SuperResolutionDataset(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            scale_factor=scale_factor,
            max_images=max_images,
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Keep 0 for reproducibility
        )

        results = []

        # Evaluate with each sampler
        for sampler_name in sampler_names:
            print(f"\nEvaluating {dataset_name} with {sampler_name}...")

            # Check if sampler exists
            if sampler_name not in self.samplers:
                print(
                    f"Error evaluating {dataset_name}: '{sampler_name}' sampler not found"
                )
                continue

            sampler_results = []
            sr_images_for_fid = []
            hr_images_for_fid = []

            try:
                for batch in tqdm(dataloader, desc=f"{sampler_name}"):
                    for i in range(len(batch["hr"])):
                        # Extract single sample from batch
                        sample = {
                            "hr": batch["hr"][i],
                            "lr": batch["lr"][i],
                            "lr_up": batch["lr_up"][i],
                            "image_name": batch["image_name"][i],
                            "dataset": batch["dataset"][i],
                            "scale_factor": batch["scale_factor"][i].item(),
                        }

                        # Evaluate single image and collect SR for FID
                        try:
                            result, sr_tensor = self.evaluate_single_image_with_sr(
                                sample=sample,
                                sampler_name=sampler_name,
                                save_images=save_images,
                                use_patches=use_patches,
                                patch_size=patch_size,
                                patch_overlap=patch_overlap,
                                patch_batch_size=patch_batch_size,
                                auto_patch_threshold=auto_patch_threshold,
                            )

                            sampler_results.append(result)

                            # Collect images for batch FID calculation
                            if len(hr_images_for_fid) < 50:  # Limit for memory
                                hr_eval = (batch["hr"][i] + 1.0) / 2.0
                                hr_images_for_fid.append(hr_eval)
                                sr_images_for_fid.append(sr_tensor)
                        except Exception as e:
                            print(
                                f"ERROR: Failed to evaluate {sample['image_name']} with {sampler_name}: {str(e)}"
                            )

                            # Create a failed result with default values
                            failed_result = {
                                "Dataset": sample["dataset"],
                                "Image": sample["image_name"],
                                "Sampler": sampler_name,
                                "Scale": sample["scale_factor"],
                                "PSNR": float("nan"),
                                "SSIM": float("nan"),
                                "MS_SSIM": float("nan"),
                                "LPIPS": float("nan"),
                                "FID": float("nan"),
                                "WSNR": float("nan"),
                                "Freq_Error": float("nan"),
                                "LR_PSNR": float("nan"),
                                "time": float("nan"),
                                "method": "",
                                "base": "",
                                "num_steps": "",
                                "error": str(e),  # Include error message for debugging
                            }
                            sampler_results.append(failed_result)
            except Exception as e:
                print(f"Error evaluating {dataset_name} with {sampler_name}: {e}")
                continue

            # Calculate batch FID
            if calculate_fid and len(hr_images_for_fid) > 1:
                try:
                    print(
                        f"Computing FID for {sampler_name}... (first run may download Inception weights)"
                    )
                    fid_score = self.metrics_calc.calculate_batch_fid(
                        hr_images_for_fid, sr_images_for_fid
                    )
                    # Update all results with FID score
                    for result in sampler_results:
                        result["FID"] = fid_score
                    print(
                        f"FID calculated successfully for {sampler_name}: {fid_score:.4f}"
                    )
                except Exception as e:
                    print(f"Error calculating FID for {sampler_name}: {e}")
                    for result in sampler_results:
                        result["FID"] = 100.0  # High FID indicates poor quality
            elif not calculate_fid:
                print(f"Skipping FID calculation for {sampler_name} (flag set)")
                for result in sampler_results:
                    result["FID"] = None
            else:
                print(f"Not enough images for FID calculation for {sampler_name}")
                for result in sampler_results:
                    result["FID"] = 100.0  # Default high FID

            results.extend(sampler_results)

        return results

    def evaluate_all_datasets(
        self,
        dataset_root: str,
        datasets: List[str] = None,
        scale_factor: int = 4,
        sampler_names: Optional[List[str]] = None,
        max_images_per_dataset: Optional[int] = None,
        save_images: bool = True,
        calculate_fid: bool = True,
        # Patch processing options
        use_patches: bool = False,
        patch_size: int = 160,
        test_crop_size: int = 2040,
        patch_overlap: int = 40,
        patch_batch_size: int = 4,
        auto_patch_threshold: int = 2048,
    ) -> Dict[str, Any]:
        """Evaluate all datasets and save comprehensive results"""
        if datasets is None:
            datasets = ["Set5", "Set14", "BSD100", "Urban100"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_results = []
        dataset_summaries = {}

        # Evaluate each dataset
        for dataset_name in datasets:
            try:
                print(f"\n{'=' * 50}")
                print(f"Evaluating dataset: {dataset_name}")
                print(f"{'=' * 50}")

                results = self.evaluate_dataset(
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    scale_factor=scale_factor,
                    sampler_names=sampler_names,
                    max_images=max_images_per_dataset,
                    save_images=save_images,
                    calculate_fid=calculate_fid,
                    # Patch processing options
                    use_patches=use_patches,
                    patch_size=patch_size,
                    patch_overlap=patch_overlap,
                    patch_batch_size=patch_batch_size,
                    auto_patch_threshold=auto_patch_threshold,
                )

                all_results.extend(results)

                # Calculate dataset summary
                dataset_summaries[dataset_name] = self._calculate_dataset_summary(
                    results
                )

            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                continue

        # Save results
        csv_path = self.output_dir / "metrics" / f"evaluation_{timestamp}.csv"
        self.metrics_calc.save_results_to_csv(all_results, str(csv_path))

        # Save experiment info
        experiment_info = {
            "timestamp": timestamp,
            "datasets": datasets,
            "scale_factor": scale_factor,
            "samplers": sampler_names or list(self.samplers.keys()),
            "total_images": len(all_results),
            "config": self.config,
            "dataset_summaries": dataset_summaries,
        }

        info_path = self.output_dir / "metrics" / f"experiment_info_{timestamp}.json"
        with open(info_path, "w") as f:
            json.dump(experiment_info, f, indent=2, default=str)

        print(f"\n{'=' * 50}")
        print(f"Evaluation completed!")
        print(f"Results saved to: {csv_path}")
        print(f"Experiment info: {info_path}")
        print(f"{'=' * 50}")

        return {
            "results": all_results,
            "summaries": dataset_summaries,
            "csv_path": str(csv_path),
            "info_path": str(info_path),
        }

    def _calculate_dataset_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for a dataset"""
        import pandas as pd

        df = pd.DataFrame(results)
        numeric_cols = [
            "PSNR",
            "SSIM",
            "MS_SSIM",
            "LPIPS",
            "WSNR",
            "Freq_Error",
            "LR_PSNR",
            "time",
        ]

        summary = {}

        # Group by sampler
        for sampler in df["Sampler"].unique():
            sampler_data = df[df["Sampler"] == sampler]
            summary[sampler] = {}

            for col in numeric_cols:
                if col in sampler_data.columns:
                    values = sampler_data[col].dropna()
                    if len(values) > 0:
                        summary[sampler][col] = {
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }

        return summary
