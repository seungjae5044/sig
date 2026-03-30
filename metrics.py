"""
Comprehensive Image Quality Metrics for Super-Resolution Evaluation
Implements 7 metrics: PSNR, SSIM, MS-SSIM, LPIPS, FID, WSNR, Freq Error
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
from typing import Dict, List, Tuple, Optional, Union
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ms_ssim
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms


class MetricsCalculator:
    """Unified calculator for all image quality metrics"""
    
    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',  # 'alex', 'vgg', 'squeeze'
        normalize_lpips: bool = True,
        fid_feature: int = 2048,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            device: Device for computation
            lpips_net: Network for LPIPS calculation
            normalize_lpips: Whether to normalize inputs for LPIPS
            fid_feature: Feature dimension for FID
            target_size: Target size for resizing (if None, use original size)
        """
        self.device = device
        self.target_size = target_size
        
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(device)
        self.normalize_lpips = normalize_lpips
        
        # Initialize FID
        self.fid_metric = FrechetInceptionDistance(
            feature=fid_feature, 
            normalize=True
        ).to(device)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def _preprocess_images(
        self,
        img_hr: torch.Tensor,
        img_sr: torch.Tensor,
        img_lr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Preprocess images for metrics calculation
        
        Args:
            img_hr: High-resolution ground truth [B, C, H, W] or [C, H, W]
            img_sr: Super-resolved image [B, C, H, W] or [C, H, W]
            img_lr: Low-resolution image [B, C, H, W] or [C, H, W] (optional)
            
        Returns:
            Preprocessed images in range [0, 1]
        """
        def process_single(img):
            if img is None:
                return None
                
            # Add batch dimension if missing
            if img.dim() == 3:
                img = img.unsqueeze(0)
            
            # Convert to float and normalize to [0, 1] if needed
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.max() > 1.0:
                img = img / 255.0
            
            # Ensure range [0, 1]
            img = torch.clamp(img, 0, 1)
            
            # Resize if target size specified
            if self.target_size is not None:
                img = F.interpolate(
                    img, size=self.target_size, mode='bilinear', align_corners=False
                )
            
            return img
        
        img_hr = process_single(img_hr)
        img_sr = process_single(img_sr)
        img_lr = process_single(img_lr) if img_lr is not None else None
        
        return img_hr, img_sr, img_lr
    
    def calculate_psnr(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # Move to CPU for skimage
        if img_hr.is_cuda:
            img_hr_np = img_hr.squeeze().cpu().numpy()
            img_sr_np = img_sr.squeeze().cpu().numpy()
        else:
            img_hr_np = img_hr.squeeze().numpy()
            img_sr_np = img_sr.squeeze().numpy()
        
        # Convert to HWC format if needed
        if img_hr_np.ndim == 3 and img_hr_np.shape[0] <= 3:
            img_hr_np = np.transpose(img_hr_np, (1, 2, 0))
            img_sr_np = np.transpose(img_sr_np, (1, 2, 0))
        
        # Calculate PSNR
        psnr_value = peak_signal_noise_ratio(img_hr_np, img_sr_np, data_range=1.0)
        return float(psnr_value)
    
    def calculate_ssim(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Structural Similarity Index"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # Move to CPU for skimage
        if img_hr.is_cuda:
            img_hr_np = img_hr.squeeze().cpu().numpy()
            img_sr_np = img_sr.squeeze().cpu().numpy()
        else:
            img_hr_np = img_hr.squeeze().numpy()
            img_sr_np = img_sr.squeeze().numpy()
        
        # Convert to HWC format if needed
        if img_hr_np.ndim == 3 and img_hr_np.shape[0] <= 3:
            img_hr_np = np.transpose(img_hr_np, (1, 2, 0))
            img_sr_np = np.transpose(img_sr_np, (1, 2, 0))
        
        # Calculate SSIM
        if img_hr_np.ndim == 3:  # Color image
            ssim_value = structural_similarity(
                img_hr_np, img_sr_np, data_range=1.0, multichannel=True, channel_axis=-1
            )
        else:  # Grayscale
            ssim_value = structural_similarity(
                img_hr_np, img_sr_np, data_range=1.0
            )
        
        return float(ssim_value)
    
    def calculate_ms_ssim(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Multi-Scale Structural Similarity Index"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # Ensure images are on same device
        img_hr = img_hr.to(self.device)
        img_sr = img_sr.to(self.device)
        
        # Calculate MS-SSIM
        ms_ssim_value = ms_ssim(img_hr, img_sr, data_range=1.0, size_average=True)
        return float(ms_ssim_value.item())
    
    def calculate_lpips(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Learned Perceptual Image Patch Similarity"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # LPIPS expects inputs in range [-1, 1]
        if self.normalize_lpips:
            img_hr = img_hr * 2.0 - 1.0
            img_sr = img_sr * 2.0 - 1.0
        
        # Ensure images are on same device
        img_hr = img_hr.to(self.device)
        img_sr = img_sr.to(self.device)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_fn(img_hr, img_sr)
        
        return float(lpips_value.mean().item())
    
    def calculate_fid_single(
        self,
        img_hr_batch: torch.Tensor,
        img_sr_batch: torch.Tensor
    ) -> float:
        """Calculate Fréchet Inception Distance for a batch of images.
        Uses per-image updates to allow variable sizes and avoids extra resizing/quantization.
        Expects float tensors in [0, 1].
        """
        try:
            img_hr_batch, img_sr_batch, _ = self._preprocess_images(img_hr_batch, img_sr_batch)

            # Ensure minimum batch size for meaningful FID calculation
            if img_hr_batch.shape[0] < 2:
                print("Warning: FID requires at least 2 images, got", img_hr_batch.shape[0])
                return 100.0

            # Reset metric
            self.fid_metric.reset()

            # Update per image to support variable sizes and avoid our own resizing
            hr = img_hr_batch.to(self.device)
            sr = img_sr_batch.to(self.device)
            for i in range(hr.shape[0]):
                self.fid_metric.update(hr[i:i+1], real=True)
            for i in range(sr.shape[0]):
                self.fid_metric.update(sr[i:i+1], real=False)

            # Compute FID
            fid_value = self.fid_metric.compute()

            # Validate
            if torch.isnan(fid_value) or torch.isinf(fid_value) or fid_value < 0:
                print(f"Warning: Invalid FID value {fid_value}, returning default")
                return 100.0

            return float(fid_value.item())

        except Exception as e:
            print(f"Error in FID calculation: {e}")
            return 100.0
    
    def calculate_wsnr(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Weighted Signal-to-Noise Ratio"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # Ensure images are on same device
        img_hr = img_hr.to(self.device)
        img_sr = img_sr.to(self.device)
        
        # Calculate MSE
        mse = F.mse_loss(img_sr, img_hr, reduction='none')
        
        # Create luminance weight map (ITU-R BT.709)
        if img_hr.shape[1] == 3:  # Color image
            weight_map = (
                0.2126 * img_hr[:, 0:1, :, :] +
                0.7152 * img_hr[:, 1:2, :, :] +
                0.0722 * img_hr[:, 2:3, :, :]
            )
        else:  # Grayscale
            weight_map = img_hr
        
        # Calculate weighted MSE
        weighted_mse = (mse * weight_map).mean()
        weighted_mse = torch.clamp(weighted_mse, min=1e-10)
        
        # Calculate WSNR
        wsnr_value = 10 * torch.log10(1.0 / weighted_mse)
        return float(wsnr_value.item())
    
    def calculate_freq_error(self, img_hr: torch.Tensor, img_sr: torch.Tensor) -> float:
        """Calculate Frequency Domain Error using high-pass filter"""
        img_hr, img_sr, _ = self._preprocess_images(img_hr, img_sr)
        
        # Ensure images are on same device
        img_hr = img_hr.to(self.device)
        img_sr = img_sr.to(self.device)
        
        # High-pass filter kernel (Laplacian)
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32, device=self.device)
        
        # Expand kernel for all channels
        c = img_hr.shape[1]
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
        
        # Apply high-pass filter
        def apply_filter(img):
            return F.conv2d(img, kernel, padding=1, groups=c)
        
        img_hr_hf = apply_filter(img_hr)
        img_sr_hf = apply_filter(img_sr)
        
        # Calculate MSE in frequency domain
        freq_error = F.mse_loss(img_sr_hf, img_hr_hf)
        return float(freq_error.item())
    
    def calculate_lr_psnr(
        self,
        img_lr: torch.Tensor,
        img_sr: torch.Tensor,
        scale_factor: int
    ) -> float:
        """Calculate LR PSNR by downsampling SR image and comparing with LR"""
        # Preprocess images properly - img_hr can be None for this calculation
        _, img_sr, img_lr = self._preprocess_images(img_sr, img_sr, img_lr)
        
        # Downsample SR image to LR size
        h_lr, w_lr = img_lr.shape[-2:]
        img_sr_downsampled = F.interpolate(
            img_sr, size=(h_lr, w_lr), mode='bicubic', align_corners=False
        )
        
        # Calculate PSNR between LR and downsampled SR
        mse = F.mse_loss(img_sr_downsampled, img_lr)
        mse = torch.clamp(mse, min=1e-10)
        psnr = 10 * torch.log10(1.0 / mse)
        
        return float(psnr.item())
    
    def calculate_all_metrics(
        self,
        img_hr: torch.Tensor,
        img_sr: torch.Tensor,
        img_lr: torch.Tensor,
        scale_factor: int = 4,
        calculate_fid: bool = False  # FID requires batch processing
    ) -> Dict[str, float]:
        """Calculate all metrics for a single image pair
        
        Args:
            img_hr: High-resolution ground truth
            img_sr: Super-resolved image
            img_lr: Low-resolution input
            scale_factor: Upsampling scale factor
            calculate_fid: Whether to calculate FID (requires batch)
            
        Returns:
            Dictionary of all metric values
        """
        results = {}
        
        try:
            results['PSNR'] = self.calculate_psnr(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            results['PSNR'] = 0.0
        
        try:
            results['SSIM'] = self.calculate_ssim(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            results['SSIM'] = 0.0
        
        try:
            results['MS_SSIM'] = self.calculate_ms_ssim(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating MS-SSIM: {e}")
            results['MS_SSIM'] = 0.0
        
        try:
            results['LPIPS'] = self.calculate_lpips(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            results['LPIPS'] = 1.0
        
        if calculate_fid:
            try:
                # Note: This should be called with batches for meaningful FID
                results['FID'] = self.calculate_fid_single(
                    img_hr.unsqueeze(0), img_sr.unsqueeze(0)
                )
            except Exception as e:
                print(f"Error calculating FID: {e}")
                results['FID'] = 100.0
        else:
            results['FID'] = None
        
        try:
            results['WSNR'] = self.calculate_wsnr(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating WSNR: {e}")
            results['WSNR'] = 0.0
        
        try:
            results['Freq_Error'] = self.calculate_freq_error(img_hr, img_sr)
        except Exception as e:
            print(f"Error calculating Freq Error: {e}")
            results['Freq_Error'] = 1.0
        
        try:
            results['LR_PSNR'] = self.calculate_lr_psnr(img_lr, img_sr, scale_factor)
        except Exception as e:
            print(f"Error calculating LR PSNR: {e}")
            results['LR_PSNR'] = 0.0
        
        return results
    
    def calculate_batch_fid(
        self,
        hr_images: List[torch.Tensor],
        sr_images: List[torch.Tensor]
    ) -> float:
        """Calculate FID for a list of images with potentially different sizes.
        Avoid manual resizing; feed per-image updates to the metric.
        """
        if len(hr_images) == 0 or len(sr_images) == 0:
            print("Warning: Empty image lists for FID calculation")
            return 100.0

        if len(hr_images) != len(sr_images):
            print(f"Warning: Mismatched image count - HR: {len(hr_images)}, SR: {len(sr_images)}")
            return 100.0

        if len(hr_images) < 2:
            print("Warning: FID requires at least 2 images, got", len(hr_images))
            return 100.0

        try:
            # Reset
            self.fid_metric.reset()

            # Update per-image (floats in [0,1])
            for img in hr_images:
                img_proc, _, _ = self._preprocess_images(img, img)
                self.fid_metric.update(img_proc.to(self.device), real=True)
            for img in sr_images:
                _, img_proc, _ = self._preprocess_images(img, img)
                self.fid_metric.update(img_proc.to(self.device), real=False)

            fid_value = self.fid_metric.compute()
            if torch.isnan(fid_value) or torch.isinf(fid_value) or fid_value < 0:
                print(f"Warning: Invalid FID value {fid_value}, returning default")
                return 100.0
            return float(fid_value.item())
        except Exception as e:
            print(f"Error in batch FID calculation: {e}")
            return 100.0
    
    def save_results_to_csv(
        self,
        results: List[Dict],
        filepath: str,
        include_summary: bool = True
    ):
        """Save evaluation results to CSV file
        
        Args:
            results: List of result dictionaries
            filepath: Output CSV file path
            include_summary: Whether to include summary statistics
        """
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if include_summary and len(df) > 1:
            # Add summary row
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary_row = {'Dataset': 'AVERAGE', 'Image': 'ALL', 'Sampler': 'ALL'}
            
            for col in numeric_cols:
                if col in df.columns:
                    # Skip None values for FID
                    non_none_values = df[col].dropna()
                    if len(non_none_values) > 0:
                        summary_row[col] = non_none_values.mean()
                    else:
                        summary_row[col] = None
            
            # Append summary
            df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df_with_summary = df
        
        # Save to CSV
        df_with_summary.to_csv(filepath, index=False, float_format='%.4f')
        print(f"Results saved to: {filepath}")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load image from file path"""
        img = Image.open(image_path).convert('RGB')
        
        # Convert to tensor
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.BICUBIC)
        
        img_tensor = transforms.ToTensor()(img)
        return img_tensor
