# -*- coding: utf-8 -*-
import torch
from pytorch_msssim import ms_ssim
from typing import Tuple

def calculate_ms_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Tính MS-SSIM sử dụng pytorch-msssim.
    
    Args:
        pred, target: Tensor shape [N, C, H, W] hoặc [C, H, W]
        
    Returns:
        MS-SSIM value (float)
    """
    # Đảm bảo input là 4D tensor
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Normalize về [0, 1] nếu cần
    if pred.max() > 1.0:
        pred = pred / 255.0
    if target.max() > 1.0:
        target = target / 255.0
    
    return ms_ssim(pred, target, data_range=1.0, size_average=True).item()

def calculate_ad(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Tính Average Distance (AD).
    
    Args:
        pred, target: Tensor shape [N, C, H, W] hoặc [C, H, W]
        
    Returns:
        AD value (float)
    """
    # Đảm bảo input là 4D tensor
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Normalize về [0, 1] nếu cần
    if pred.max() > 1.0:
        pred = pred / 255.0
    if target.max() > 1.0:
        target = target / 255.0
    
    # Tính L2 distance trung bình
    diff = torch.sqrt(torch.sum((pred - target) ** 2, dim=1) / pred.shape[1])
    return torch.mean(diff).item()

def evaluate_batch(pred_images: torch.Tensor, target_images: torch.Tensor) -> Tuple[float, float]:
    """
    Đánh giá batch với MS-SSIM và AD.
    
    Returns:
        (ms_ssim_score, ad_score)
    """
    ms_ssim_score = calculate_ms_ssim(pred_images, target_images)
    ad_score = calculate_ad(pred_images, target_images)
    return ms_ssim_score, ad_score

# Ví dụ sử dụng
if __name__ == "__main__":
    # Test
    pred = torch.rand(2, 3, 256, 256)
    target = pred + 0.1 * torch.randn_like(pred)
    
    ms_ssim_score, ad_score = evaluate_batch(pred, target)
    print(f"MS-SSIM: {ms_ssim_score:.4f}")
    print(f"AD: {ad_score:.4f}")