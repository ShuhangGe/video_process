"""
Video Tokenizer - Calculate token counts from frame dimensions and metadata.

This module provides token calculation functionality for video processing,
helping estimate model input requirements and optimize processing parameters.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch


logger = logging.getLogger(__name__)


class VideoTokenizer:
    """
    Video tokenizer for calculating token counts from video characteristics.
    
    Provides:
    - Token count estimation from frame dimensions
    - Grid calculations for multi-dimensional positional encoding
    - Memory and computational cost estimation
    - Model-specific token optimization
    """
    
    def __init__(self, config):
        """Initialize video tokenizer with configuration."""
        self.config = config
        self.processing_config = config.processing
        
        # Token calculation parameters
        self.image_factor = self.processing_config.image_factor
        self.frame_factor = config.sampling.frame_factor
        
        logger.debug(f"VideoTokenizer initialized with image_factor={self.image_factor}")
    
    def calculate_tokens(self, video_tensor: torch.Tensor, video_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate token information for a video tensor.
        
        Args:
            video_tensor: Video tensor (T, C, H, W)
            video_config: Video configuration dictionary
            
        Returns:
            Dictionary with token information
        """
        if video_tensor.numel() == 0:
            return self._empty_token_info()
        
        nframes, channels, height, width = video_tensor.shape
        
        # Calculate basic token counts
        tokens_per_frame = self._calculate_tokens_per_frame(height, width)
        total_video_tokens = tokens_per_frame * nframes
        
        # Calculate grid dimensions for positional encoding
        grid_t, grid_h, grid_w = self._calculate_grid_dimensions(nframes, height, width)
        
        # Calculate memory requirements
        memory_info = self._estimate_memory_requirements(total_video_tokens, video_tensor)
        
        # Get model-specific information if available
        model_info = self._get_model_specific_info(total_video_tokens, video_config)
        
        token_info = {
            # Basic token information
            "total_tokens": total_video_tokens,
            "tokens_per_frame": tokens_per_frame,
            "num_frames": nframes,
            
            # Dimensions
            "height": height,
            "width": width,
            "channels": channels,
            
            # Grid information for positional encoding
            "grid_t": grid_t,
            "grid_h": grid_h, 
            "grid_w": grid_w,
            "grid_thw": [grid_t, grid_h, grid_w],
            
            # Memory estimation
            "memory_info": memory_info,
            
            # Model-specific information
            "model_info": model_info,
            
            # Token density metrics
            "tokens_per_pixel": total_video_tokens / (nframes * height * width),
            "compression_ratio": (nframes * height * width) / total_video_tokens,
            
            # Processing metadata
            "image_factor": self.image_factor,
            "frame_factor": self.frame_factor,
        }
        
        logger.debug(
            f"Token calculation: {nframes}x{height}x{width} -> {total_video_tokens} tokens "
            f"(grid: {grid_t}x{grid_h}x{grid_w})"
        )
        
        return token_info
    
    def _calculate_tokens_per_frame(self, height: int, width: int) -> int:
        """Calculate tokens per frame based on dimensions."""
        # Each patch of image_factor x image_factor becomes one token
        tokens_h = height // self.image_factor
        tokens_w = width // self.image_factor
        
        tokens_per_frame = tokens_h * tokens_w
        
        logger.debug(
            f"Tokens per frame: {height}x{width} -> {tokens_h}x{tokens_w} = {tokens_per_frame} tokens"
        )
        
        return tokens_per_frame
    
    def _calculate_grid_dimensions(self, nframes: int, height: int, width: int) -> Tuple[int, int, int]:
        """Calculate grid dimensions for 3D positional encoding."""
        # Temporal dimension (number of frames)
        grid_t = nframes
        
        # Spatial dimensions (based on tokenization)
        grid_h = height // self.image_factor
        grid_w = width // self.image_factor
        
        return grid_t, grid_h, grid_w
    
    def _estimate_memory_requirements(self, total_tokens: int, video_tensor: torch.Tensor) -> Dict[str, float]:
        """Estimate memory requirements for token processing."""
        # Token embedding memory (assume 4096 hidden dimensions)
        hidden_dim = 4096  # Common hidden dimension
        token_memory_bytes = total_tokens * hidden_dim * 4  # float32
        
        # Attention memory (quadratic in sequence length)
        attention_memory_bytes = total_tokens * total_tokens * 4  # Simplified estimate
        
        # Input video memory
        input_memory_bytes = video_tensor.numel() * video_tensor.element_size()
        
        # Total estimated memory
        total_memory_bytes = token_memory_bytes + attention_memory_bytes + input_memory_bytes
        
        return {
            "token_memory_mb": token_memory_bytes / (1024 * 1024),
            "attention_memory_mb": attention_memory_bytes / (1024 * 1024),
            "input_memory_mb": input_memory_bytes / (1024 * 1024),
            "total_memory_mb": total_memory_bytes / (1024 * 1024),
            "memory_per_token_kb": (token_memory_bytes + attention_memory_bytes) / total_tokens / 1024,
        }
    
    def _get_model_specific_info(self, total_tokens: int, video_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model-specific token information."""
        model_info = {
            "total_tokens": total_tokens,
            "is_within_limits": True,
            "recommendations": []
        }
        

        # General recommendations based on token count
        if total_tokens > 100000:
            model_info["recommendations"].append(
                "Very high token count. Consider using frame sampling or lower resolution."
            )
        elif total_tokens < 1000:
            model_info["recommendations"].append(
                "Low token count. Consider higher resolution or more frames for better quality."
            )
        
        return model_info
    
    def _empty_token_info(self) -> Dict[str, Any]:
        """Return token info for empty video."""
        return {
            "total_tokens": 0,
            "tokens_per_frame": 0,
            "num_frames": 0,
            "height": 0,
            "width": 0,
            "channels": 0,
            "grid_t": 0,
            "grid_h": 0,
            "grid_w": 0,
            "grid_thw": [0, 0, 0],
            "memory_info": {
                "token_memory_mb": 0.0,
                "attention_memory_mb": 0.0,
                "input_memory_mb": 0.0,
                "total_memory_mb": 0.0,
                "memory_per_token_kb": 0.0,
            },
            "model_info": {
                "total_tokens": 0,
                "is_within_limits": True,
                "recommendations": ["Empty video - no tokens calculated"]
            },
            "tokens_per_pixel": 0.0,
            "compression_ratio": float('inf'),
            "image_factor": self.image_factor,
            "frame_factor": self.frame_factor,
        }
    
    def optimize_for_token_limit(self, video_config: Dict[str, Any], 
                                target_tokens: int) -> Dict[str, Any]:
        """
        Optimize video configuration to meet token limit.
        
        Args:
            video_config: Original video configuration
            target_tokens: Target token count
            
        Returns:
            Optimized video configuration
        """
        # Start with current config
        optimized_config = video_config.copy()
        
        # Get current estimated tokens (rough calculation)
        current_height = video_config.get("resized_height", 224)
        current_width = video_config.get("resized_width", 224)
        current_frames = video_config.get("nframes", 16)
        
        current_tokens = self._calculate_tokens_per_frame(current_height, current_width) * current_frames
        
        if current_tokens <= target_tokens:
            optimized_config["optimization_applied"] = False
            return optimized_config
        
        # Calculate reduction factor needed
        reduction_factor = math.sqrt(target_tokens / current_tokens)
        
        # Reduce dimensions
        new_height = max(self.image_factor, int(current_height * reduction_factor // self.image_factor) * self.image_factor)
        new_width = max(self.image_factor, int(current_width * reduction_factor // self.image_factor) * self.image_factor)
        
        # Check if dimension reduction is enough
        new_tokens_per_frame = self._calculate_tokens_per_frame(new_height, new_width)
        new_total_tokens = new_tokens_per_frame * current_frames
        
        if new_total_tokens > target_tokens:
            # Also need to reduce frame count
            max_frames = target_tokens // new_tokens_per_frame
            max_frames = max(self.frame_factor, (max_frames // self.frame_factor) * self.frame_factor)
            optimized_config["nframes"] = max_frames
        
        optimized_config.update({
            "resized_height": new_height,
            "resized_width": new_width,
            "optimization_applied": True,
            "original_tokens": current_tokens,
            "target_tokens": target_tokens,
            "estimated_tokens": self._calculate_tokens_per_frame(new_height, new_width) * optimized_config.get("nframes", current_frames),
            "reduction_factor": reduction_factor,
        })
        
        logger.info(
            f"Token optimization: {current_tokens} -> {optimized_config['estimated_tokens']} tokens, "
            f"dimensions: {current_height}x{current_width} -> {new_height}x{new_width}, "
            f"frames: {current_frames} -> {optimized_config.get('nframes', current_frames)}"
        )
        
        return optimized_config
    
    def calculate_batch_tokens(self, video_tensors: List[torch.Tensor], 
                             video_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate token information for a batch of videos.
        
        Args:
            video_tensors: List of video tensors
            video_configs: List of video configurations
            
        Returns:
            Batch token information
        """
        if len(video_tensors) != len(video_configs):
            raise ValueError("Number of video tensors must match number of configs")
        
        batch_info = {
            "batch_size": len(video_tensors),
            "individual_tokens": [],
            "total_batch_tokens": 0,
            "max_tokens": 0,
            "min_tokens": float('inf'),
            "avg_tokens": 0.0,
            "memory_info": {
                "total_memory_mb": 0.0,
                "max_memory_mb": 0.0,
                "avg_memory_mb": 0.0,
            }
        }
        
        if not video_tensors:
            batch_info["min_tokens"] = 0
            return batch_info
        
        total_tokens = 0
        total_memory = 0.0
        max_memory = 0.0
        
        for video_tensor, video_config in zip(video_tensors, video_configs):
            token_info = self.calculate_tokens(video_tensor, video_config)
            
            tokens = token_info["total_tokens"]
            memory = token_info["memory_info"]["total_memory_mb"]
            
            batch_info["individual_tokens"].append(token_info)
            total_tokens += tokens
            total_memory += memory
            
            batch_info["max_tokens"] = max(batch_info["max_tokens"], tokens)
            batch_info["min_tokens"] = min(batch_info["min_tokens"], tokens)
            max_memory = max(max_memory, memory)
        
        batch_info["total_batch_tokens"] = total_tokens
        batch_info["avg_tokens"] = total_tokens / len(video_tensors)
        batch_info["memory_info"].update({
            "total_memory_mb": total_memory,
            "max_memory_mb": max_memory,
            "avg_memory_mb": total_memory / len(video_tensors),
        })
        
        logger.info(
            f"Batch token calculation: {len(video_tensors)} videos, "
            f"total: {total_tokens} tokens, avg: {batch_info['avg_tokens']:.1f} tokens/video"
        )
        
        return batch_info
    
    def estimate_processing_time(self, total_tokens: int, model_name: str = None) -> Dict[str, float]:
        """
        Estimate processing time based on token count.
        
        Args:
            total_tokens: Total token count
            model_name: Optional model name for specific estimates
            
        Returns:
            Processing time estimates
        """
        # Base processing rates (tokens per second) - rough estimates
        base_rates = {
            "qwen2.5-vl-7b": 1000,   # tokens/sec
            "qwen2.5-vl-32b": 400,   # tokens/sec
            "qwen2.5-vl-72b": 200,   # tokens/sec
            "default": 500,          # tokens/sec
        }
        
        # Determine processing rate
        if model_name:
            model_key = model_name.lower()
            rate = None
            for key in base_rates:
                if key in model_key and key != "default":
                    rate = base_rates[key]
                    break
            if rate is None:
                rate = base_rates["default"]
        else:
            rate = base_rates["default"]
        
        # Calculate estimates
        base_time = total_tokens / rate
        
        # Add overhead for attention computation (quadratic scaling)
        attention_overhead = (total_tokens / 10000) ** 2 * 0.1  # Simplified model
        
        total_time = base_time + attention_overhead
        
        return {
            "base_processing_time_sec": base_time,
            "attention_overhead_sec": attention_overhead,
            "total_estimated_time_sec": total_time,
            "tokens_per_second": rate,
            "model_used": model_name or "default",
        }
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about tokenizer configuration."""
        return {
            "image_factor": self.image_factor,
            "frame_factor": self.frame_factor,
            "tokens_per_patch": 1,  # Each patch becomes one token
            "patch_size": f"{self.image_factor}x{self.image_factor}",
            "supports_3d_encoding": True,
            "max_sequence_length": None,
        }


def calculate_tokens_simple(video_tensor: torch.Tensor, image_factor: int = 28) -> int:
    """
    Simple function to calculate total tokens from video tensor.
    
    Args:
        video_tensor: Video tensor (T, C, H, W)
        image_factor: Image tokenization factor
        
    Returns:
        Total token count
    """
    if video_tensor.numel() == 0:
        return 0
    
    nframes, _, height, width = video_tensor.shape
    tokens_per_frame = (height // image_factor) * (width // image_factor)
    return tokens_per_frame * nframes


def estimate_optimal_dimensions(target_tokens: int, nframes: int, image_factor: int = 28) -> Tuple[int, int]:
    """
    Estimate optimal height and width for target token count.
    
    Args:
        target_tokens: Target total token count
        nframes: Number of frames
        image_factor: Image tokenization factor
        
    Returns:
        Tuple of (height, width)
    """
    tokens_per_frame = target_tokens / max(nframes, 1)
    
    # Assume square frames for simplicity
    tokens_per_side = math.sqrt(tokens_per_frame)
    pixels_per_side = tokens_per_side * image_factor
    
    # Round to image_factor boundaries
    height = max(image_factor, int(pixels_per_side // image_factor) * image_factor)
    width = height
    
    return height, width 