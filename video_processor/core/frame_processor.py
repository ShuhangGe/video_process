"""
Frame Processor - Resize and process extracted frames for model consumption.

This module handles frame resizing, normalization, and other preprocessing operations
to prepare frames for multimodal AI models while maintaining aspect ratios and quality.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


logger = logging.getLogger(__name__)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_ratio = 200  # Maximum aspect ratio allowed
    
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}"
        )
    
    # Round to nearest factor
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    
    if h_bar * w_bar > max_pixels:
        # Scale down to fit max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        # Scale up to meet min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar


class FrameProcessor:
    """
    Frame processor for resizing and preprocessing video frames.
    
    Handles:
    - Smart resizing with aspect ratio preservation
    - Multiple interpolation modes
    - Batch processing
    - Memory management
    - Quality optimization
    """
    
    def __init__(self, config):
        """Initialize frame processor with configuration."""
        self.config = config
        self.processing_config = config.processing
        
        # Setup transforms
        self._setup_transforms()
        
        logger.debug(f"FrameProcessor initialized with {self.processing_config.interpolation_mode} interpolation")
    
    def _setup_transforms(self):
        """Setup image transforms based on configuration."""
        # Base transforms
        self.to_tensor = transforms.ToTensor()
        
        # Interpolation mode
        interp_mode = getattr(InterpolationMode, self.processing_config.interpolation_mode.upper())
        
        # Create resize transform (will be dynamically configured)
        self.resize_transform = lambda size: transforms.Resize(
            size, 
            interpolation=interp_mode,
            antialias=self.processing_config.antialias
        )
        
        # Normalization transform (optional)
        if self.processing_config.normalize:
            # Standard ImageNet normalization
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize_transform = None
    
    def process_frames(self, video_tensor: torch.Tensor, video_config: Dict[str, Any]) -> torch.Tensor:
        """
        Process video frames with resizing and preprocessing.
        
        Args:
            video_tensor: Input video tensor (T, C, H, W)
            video_config: Video configuration dictionary
            
        Returns:
            Processed video tensor with same shape format
        """
        if video_tensor.numel() == 0:
            logger.warning("Empty video tensor received")
            return video_tensor
        
        original_shape = video_tensor.shape
        nframes, channels, height, width = original_shape
        
        logger.debug(f"Processing {nframes} frames of shape {original_shape}")
        
        # Determine target dimensions
        target_height, target_width = self._calculate_target_dimensions(
            height, width, video_config, nframes
        )
        
        # Process frames
        if target_height == height and target_width == width:
            # No resizing needed
            processed_tensor = video_tensor
        else:
            # Resize frames
            processed_tensor = self._resize_frames(
                video_tensor, target_height, target_width
            )
        
        # Apply additional processing
        processed_tensor = self._apply_post_processing(processed_tensor, video_config)
        
        logger.info(
            f"Processed frames: {original_shape} -> {processed_tensor.shape}, "
            f"target: {target_height}x{target_width}"
        )
        
        return processed_tensor
    
    def _calculate_target_dimensions(self, height: int, width: int, 
                                   video_config: Dict[str, Any], nframes: int) -> Tuple[int, int]:
        """Calculate target dimensions for frame resizing."""
        # Check for explicit dimensions in config
        if "resized_height" in video_config and "resized_width" in video_config:
            target_height = smart_resize(
                video_config["resized_height"],
                video_config["resized_width"],
                factor=self.processing_config.image_factor,
            )[0]
            target_width = smart_resize(
                video_config["resized_height"],
                video_config["resized_width"],
                factor=self.processing_config.image_factor,
            )[1]
            return target_height, target_width
        
        # Calculate dynamic pixel constraints for videos
        min_pixels = video_config.get("min_pixels", self.processing_config.video_min_pixels)
        
        # Calculate max pixels based on total pixel budget
        total_pixels = video_config.get("total_pixels", self.processing_config.video_total_pixels)
        max_pixels_per_frame = max(
            min(self.processing_config.video_max_pixels, total_pixels // max(nframes, 1) * 2),
            int(min_pixels * 1.05)
        )
        
        # Apply smart resize
        target_height, target_width = smart_resize(
            height,
            width,
            factor=self.processing_config.image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels_per_frame,
        )
        
        logger.debug(
            f"Calculated target dimensions: {height}x{width} -> {target_height}x{target_width}, "
            f"pixels: {height*width} -> {target_height*target_width}, "
            f"constraints: {min_pixels}-{max_pixels_per_frame}"
        )
        
        return target_height, target_width
    
    def _resize_frames(self, video_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize frames to target dimensions."""
        # Use torchvision's functional resize for efficiency
        resized_tensor = transforms.functional.resize(
            video_tensor,
            [target_height, target_width],
            interpolation=getattr(InterpolationMode, self.processing_config.interpolation_mode.upper()),
            antialias=self.processing_config.antialias,
        )
        
        return resized_tensor.float()
    
    def _apply_post_processing(self, video_tensor: torch.Tensor, video_config: Dict[str, Any]) -> torch.Tensor:
        """Apply additional post-processing operations."""
        processed = video_tensor
        
        # Apply normalization if configured
        if self.normalize_transform is not None:
            # Apply normalization frame by frame
            normalized_frames = []
            for frame in processed:
                normalized_frame = self.normalize_transform(frame)
                normalized_frames.append(normalized_frame)
            processed = torch.stack(normalized_frames, dim=0)
        
        # Ensure correct dtype
        if self.processing_config.enable_half_precision:
            processed = processed.half()
        else:
            processed = processed.float()
        
        # Clamp values to valid range
        processed = torch.clamp(processed, 0.0, 1.0)
        
        return processed
    
    def process_batch(self, video_tensors: List[torch.Tensor], 
                     video_configs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Process multiple video tensors in batch for efficiency.
        
        Args:
            video_tensors: List of video tensors
            video_configs: List of corresponding video configurations
            
        Returns:
            List of processed video tensors
        """
        if len(video_tensors) != len(video_configs):
            raise ValueError("Number of video tensors must match number of configs")
        
        if not video_tensors:
            return []
        
        batch_size = min(len(video_tensors), self.processing_config.max_batch_size)
        processed_videos = []
        
        for i in range(0, len(video_tensors), batch_size):
            batch_videos = video_tensors[i:i + batch_size]
            batch_configs = video_configs[i:i + batch_size]
            
            batch_processed = []
            for video_tensor, video_config in zip(batch_videos, batch_configs):
                processed = self.process_frames(video_tensor, video_config)
                batch_processed.append(processed)
            
            processed_videos.extend(batch_processed)
        
        logger.info(f"Batch processed {len(video_tensors)} videos")
        return processed_videos
    
    def estimate_memory_usage(self, video_tensor: torch.Tensor, target_height: int, target_width: int) -> Dict[str, float]:
        """Estimate memory usage for processing."""
        nframes, channels, height, width = video_tensor.shape
        
        # Input memory (bytes)
        input_memory = video_tensor.numel() * video_tensor.element_size()
        
        # Output memory (bytes)
        output_elements = nframes * channels * target_height * target_width
        if self.processing_config.enable_half_precision:
            output_memory = output_elements * 2  # half precision = 2 bytes
        else:
            output_memory = output_elements * 4  # float32 = 4 bytes
        
        # Peak memory (during processing)
        peak_memory = input_memory + output_memory + (output_memory * 0.1)  # 10% overhead
        
        return {
            "input_memory_mb": input_memory / (1024 * 1024),
            "output_memory_mb": output_memory / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "compression_ratio": input_memory / max(output_memory, 1),
        }
    
    def optimize_for_memory(self, video_tensor: torch.Tensor, memory_limit_mb: float = 1024) -> Dict[str, Any]:
        """
        Optimize processing parameters for memory constraints.
        
        Args:
            video_tensor: Input video tensor
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Optimized configuration parameters
        """
        nframes, channels, height, width = video_tensor.shape
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Calculate current memory usage
        current_memory = video_tensor.numel() * video_tensor.element_size()
        
        if current_memory <= memory_limit_bytes:
            return {"optimized": False, "original_config": True}
        
        # Calculate reduction factor needed
        reduction_factor = math.sqrt(memory_limit_bytes / current_memory * 0.8)  # 80% of limit for safety
        
        # Adjust dimensions
        new_height = max(28, int(height * reduction_factor // 28) * 28)
        new_width = max(28, int(width * reduction_factor // 28) * 28)
        
        # Consider frame reduction if still too large
        new_memory = nframes * channels * new_height * new_width * 4  # float32
        if new_memory > memory_limit_bytes:
            frame_reduction = memory_limit_bytes / new_memory * 0.8
            max_frames = max(4, int(nframes * frame_reduction // 2) * 2)
        else:
            max_frames = nframes
        
        optimization_config = {
            "optimized": True,
            "original_shape": (height, width),
            "optimized_shape": (new_height, new_width),
            "original_frames": nframes,
            "max_frames": max_frames,
            "reduction_factor": reduction_factor,
            "estimated_memory_mb": new_memory / (1024 * 1024),
            "memory_limit_mb": memory_limit_mb,
        }
        
        logger.info(
            f"Memory optimization: {height}x{width} -> {new_height}x{new_width}, "
            f"frames: {nframes} -> {max_frames}, "
            f"memory: {current_memory/(1024*1024):.1f}MB -> {new_memory/(1024*1024):.1f}MB"
        )
        
        return optimization_config
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about current processing configuration."""
        return {
            "image_factor": self.processing_config.image_factor,
            "interpolation_mode": self.processing_config.interpolation_mode,
            "antialias": self.processing_config.antialias,
            "normalize": self.processing_config.normalize,
            "half_precision": self.processing_config.enable_half_precision,
            "max_batch_size": self.processing_config.max_batch_size,
            "min_pixels": self.processing_config.min_pixels,
            "max_pixels": self.processing_config.max_pixels,
            "video_min_pixels": self.processing_config.video_min_pixels,
            "video_max_pixels": self.processing_config.video_max_pixels,
            "video_total_pixels": self.processing_config.video_total_pixels,
        }


class AdvancedFrameProcessor(FrameProcessor):
    """
    Advanced frame processor with additional features like quality assessment and adaptive processing.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._quality_cache = {}
    
    def assess_frame_quality(self, video_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Assess the quality of video frames.
        
        Args:
            video_tensor: Video tensor (T, C, H, W)
            
        Returns:
            Dictionary with quality metrics
        """
        if video_tensor.numel() == 0:
            return {"overall_quality": 0.0, "sharpness": 0.0, "contrast": 0.0}
        
        # Calculate sharpness using Laplacian variance
        sharpness_scores = []
        contrast_scores = []
        
        for frame in video_tensor:
            # Convert to grayscale for analysis
            gray_frame = torch.mean(frame, dim=0, keepdim=True)
            
            # Sharpness (Laplacian variance)
            laplacian_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float()
            laplacian = torch.nn.functional.conv2d(gray_frame.unsqueeze(0), laplacian_kernel, padding=1)
            sharpness = torch.var(laplacian).item()
            sharpness_scores.append(sharpness)
            
            # Contrast (standard deviation of pixel values)
            contrast = torch.std(gray_frame).item()
            contrast_scores.append(contrast)
        
        # Calculate overall quality metrics
        avg_sharpness = sum(sharpness_scores) / len(sharpness_scores)
        avg_contrast = sum(contrast_scores) / len(contrast_scores)
        
        # Normalize and combine (simplified quality model)
        normalized_sharpness = min(avg_sharpness / 100.0, 1.0)  # Rough normalization
        normalized_contrast = min(avg_contrast / 0.3, 1.0)  # Rough normalization
        
        overall_quality = (normalized_sharpness * 0.6 + normalized_contrast * 0.4)
        
        return {
            "overall_quality": overall_quality,
            "sharpness": normalized_sharpness,
            "contrast": normalized_contrast,
            "frame_scores": list(zip(sharpness_scores, contrast_scores))
        }
    
    def adaptive_resize(self, video_tensor: torch.Tensor, video_config: Dict[str, Any]) -> torch.Tensor:
        """
        Adaptive resizing based on content analysis.
        
        Args:
            video_tensor: Input video tensor
            video_config: Video configuration
            
        Returns:
            Adaptively resized video tensor
        """
        # Assess quality to determine optimal resize strategy
        quality_info = self.assess_frame_quality(video_tensor)
        
        # Adjust resize parameters based on quality
        if quality_info["overall_quality"] > 0.8:
            # High quality: can afford more aggressive compression
            quality_factor = 0.9
        elif quality_info["overall_quality"] > 0.5:
            # Medium quality: standard processing
            quality_factor = 1.0
        else:
            # Low quality: preserve more pixels
            quality_factor = 1.2
        
        # Modify config for adaptive processing
        adaptive_config = video_config.copy()
        
        if "min_pixels" in adaptive_config:
            adaptive_config["min_pixels"] = int(adaptive_config["min_pixels"] * quality_factor)
        if "max_pixels" in adaptive_config:
            adaptive_config["max_pixels"] = int(adaptive_config["max_pixels"] * quality_factor)
        
        logger.debug(
            f"Adaptive resize: quality={quality_info['overall_quality']:.3f}, "
            f"factor={quality_factor:.2f}"
        )
        
        return self.process_frames(video_tensor, adaptive_config)


def process_frames_simple(video_tensor: torch.Tensor, target_height: int = None, 
                         target_width: int = None) -> torch.Tensor:
    """
    Simple function to process video frames.
    
    Args:
        video_tensor: Input video tensor (T, C, H, W)
        target_height: Target height (optional)
        target_width: Target width (optional)
        
    Returns:
        Processed video tensor
    """
    from ..config import get_default_config
    
    processor = FrameProcessor(get_default_config())
    video_config = {}
    
    if target_height is not None and target_width is not None:
        video_config["resized_height"] = target_height
        video_config["resized_width"] = target_width
    
    return processor.process_frames(video_tensor, video_config) 