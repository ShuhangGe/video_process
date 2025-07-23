"""
Smart Sampler - Intelligent frame sampling based on video characteristics and model requirements.

This module implements various sampling strategies to optimize frame selection for
multimodal AI models while maintaining temporal coherence and important visual information.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache

import torch


logger = logging.getLogger(__name__)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class SmartSampler:
    """
    Intelligent frame sampler that adapts to video characteristics.
    
    Supports multiple sampling strategies:
    - Uniform: Even distribution across video timeline
    - Adaptive: Dynamic sampling based on scene complexity
    - FPS-based: Target specific frames per second output
    - Duration-based: Sample based on video length intervals
    - Keyframe: Prefer keyframes when available
    """
    
    def __init__(self, config):
        """Initialize smart sampler with configuration."""
        self.config = config
        self.sampling_config = config.sampling
        
        logger.debug(f"SmartSampler initialized with strategy: {self.sampling_config.strategy}")
    
    def calculate_nframes(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """
        Calculate the number of frames to sample based on configuration and video characteristics.
        
        Args:
            video_config: Video configuration dictionary
            total_frames: Total number of frames in the video
            video_fps: Original video FPS
            
        Returns:
            Number of frames to sample
        """
        # Handle explicit nframes specification
        if "nframes" in video_config:
            nframes = round_by_factor(video_config["nframes"], self.sampling_config.frame_factor)
            return self._validate_nframes(nframes, total_frames)
        
        # Use sampling strategy
        strategy = video_config.get("strategy", self.sampling_config.strategy)
        
        if strategy == "uniform":
            return self._uniform_sampling(video_config, total_frames, video_fps)
        elif strategy == "adaptive":
            return self._adaptive_sampling(video_config, total_frames, video_fps)
        elif strategy == "keyframe":
            return self._keyframe_sampling(video_config, total_frames, video_fps)
        elif strategy == "duration":
            return self._duration_based_sampling(video_config, total_frames, video_fps)
        else:
            logger.warning(f"Unknown sampling strategy: {strategy}, falling back to uniform")
            return self._uniform_sampling(video_config, total_frames, video_fps)
    
    def _uniform_sampling(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """Uniform sampling across the video timeline."""
        # Get FPS parameters
        target_fps = video_config.get("fps", self.sampling_config.target_fps)
        min_frames = ceil_by_factor(
            video_config.get("min_frames", self.sampling_config.min_frames), 
            self.sampling_config.frame_factor
        )
        max_frames = floor_by_factor(
            video_config.get("max_frames", min(self.sampling_config.max_frames, total_frames)), 
            self.sampling_config.frame_factor
        )
        
        # Calculate target frames based on FPS
        video_duration = total_frames / video_fps if video_fps > 0 else 1.0
        target_nframes = video_duration * target_fps
        
        # Apply constraints
        nframes = max(min_frames, min(target_nframes, max_frames))
        nframes = min(nframes, total_frames)
        nframes = floor_by_factor(nframes, self.sampling_config.frame_factor)
        
        logger.debug(
            f"Uniform sampling: video_duration={video_duration:.2f}s, "
            f"target_fps={target_fps}, target_nframes={target_nframes:.1f}, "
            f"final_nframes={nframes}"
        )
        
        return int(nframes)
    
    def _adaptive_sampling(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """Adaptive sampling based on video duration and content complexity."""
        video_duration = total_frames / video_fps if video_fps > 0 else 1.0
        
        # Base uniform sampling
        base_nframes = self._uniform_sampling(video_config, total_frames, video_fps)
        
        # Adaptive adjustments based on video characteristics
        if video_duration <= self.sampling_config.short_video_threshold:
            # Short videos: sample more frames relative to duration
            multiplier = 1.5
        elif video_duration >= self.sampling_config.long_video_threshold:
            # Long videos: reduce frame density to focus on key moments
            multiplier = 0.8
        else:
            # Medium videos: standard sampling
            multiplier = 1.0
        
        # Apply scene change considerations (simplified heuristic)
        # In a real implementation, this could analyze frame differences
        scene_complexity_factor = 1.0  # Placeholder for actual scene analysis
        
        adjusted_nframes = base_nframes * multiplier * scene_complexity_factor
        adjusted_nframes = floor_by_factor(adjusted_nframes, self.sampling_config.frame_factor)
        adjusted_nframes = max(self.sampling_config.min_frames, 
                              min(adjusted_nframes, self.sampling_config.max_frames))
        adjusted_nframes = min(adjusted_nframes, total_frames)
        
        logger.debug(
            f"Adaptive sampling: duration={video_duration:.2f}s, "
            f"base_nframes={base_nframes}, multiplier={multiplier:.2f}, "
            f"final_nframes={adjusted_nframes}"
        )
        
        return int(adjusted_nframes)
    
    def _keyframe_sampling(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """Keyframe-based sampling (simplified - prefers uniform distribution for now)."""
        # In a full implementation, this would:
        # 1. Detect keyframes in the video
        # 2. Prioritize sampling at or near keyframes
        # 3. Fill gaps with uniform sampling
        
        # For now, use uniform sampling with slight preference for more frames
        base_nframes = self._uniform_sampling(video_config, total_frames, video_fps)
        
        # Slightly increase frame count to catch more keyframes
        keyframe_nframes = min(
            floor_by_factor(base_nframes * 1.2, self.sampling_config.frame_factor),
            self.sampling_config.max_frames,
            total_frames
        )
        
        logger.debug(
            f"Keyframe sampling: base_nframes={base_nframes}, "
            f"keyframe_nframes={keyframe_nframes}"
        )
        
        return int(keyframe_nframes)
    
    def _duration_based_sampling(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """Duration-based sampling using base interval."""
        video_duration = total_frames / video_fps if video_fps > 0 else 1.0
        base_interval = video_config.get("base_interval", self.sampling_config.base_interval)
        
        # Calculate frames based on interval
        interval_nframes = math.ceil(video_duration / base_interval)
        interval_nframes = round_by_factor(interval_nframes, self.sampling_config.frame_factor)
        
        # Apply min/max constraints
        interval_nframes = max(self.sampling_config.min_frames, interval_nframes)
        interval_nframes = min(self.sampling_config.max_frames, interval_nframes)
        interval_nframes = min(interval_nframes, total_frames)
        
        logger.debug(
            f"Duration-based sampling: duration={video_duration:.2f}s, "
            f"interval={base_interval}s, interval_nframes={interval_nframes}"
        )
        
        return int(interval_nframes)
    
    def _validate_nframes(self, nframes: int, total_frames: int) -> int:
        """Validate and adjust nframes to be within acceptable bounds."""
        if total_frames == 0:
            logger.warning("Video has no frames, returning minimum frame count")
            return self.sampling_config.frame_factor
        
        # Ensure nframes is within valid range
        min_allowed = self.sampling_config.frame_factor
        max_allowed = total_frames
        
        if not (min_allowed <= nframes <= max_allowed):
            original_nframes = nframes
            nframes = max(min_allowed, min(nframes, max_allowed))
            logger.warning(
                f"Adjusted nframes from {original_nframes} to {nframes} "
                f"(valid range: {min_allowed}-{max_allowed})"
            )
        
        return nframes
    
    def calculate_sampling_indices(self, nframes: int, total_frames: int, 
                                 start_frame: int = 0, end_frame: Optional[int] = None) -> torch.Tensor:
        """
        Calculate the actual frame indices to sample.
        
        Args:
            nframes: Number of frames to sample
            total_frames: Total frames available
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            
        Returns:
            Tensor of frame indices to sample
        """
        if end_frame is None:
            end_frame = total_frames - 1
        
        if start_frame >= end_frame:
            raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")
        
        # Generate indices using linear interpolation
        indices = torch.linspace(start_frame, end_frame, nframes).round().long()
        
        # Ensure indices are unique and within bounds
        indices = torch.clamp(indices, start_frame, end_frame)
        
        # Remove duplicates while maintaining order
        indices = self._remove_duplicate_indices(indices)
        
        logger.debug(f"Generated {len(indices)} sampling indices from {start_frame} to {end_frame}")
        
        return indices
    
    def _remove_duplicate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Remove duplicate indices while maintaining order."""
        if len(indices) <= 1:
            return indices
        
        # Use torch.unique with return_inverse to maintain order
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
        
        # If we lost indices due to duplicates, redistribute
        if len(unique_indices) < len(indices):
            logger.debug(f"Removed {len(indices) - len(unique_indices)} duplicate indices")
            
            # Redistribute to maintain desired frame count
            start_idx = indices[0].item()
            end_idx = indices[-1].item()
            redistributed = torch.linspace(start_idx, end_idx, len(indices)).round().long()
            return redistributed
        
        return indices
    
    def get_sampling_info(self, video_config: Dict[str, Any], total_frames: int, 
                         video_fps: float) -> Dict[str, Any]:
        """
        Get detailed information about the sampling strategy and parameters.
        
        Args:
            video_config: Video configuration
            total_frames: Total frames in video
            video_fps: Video FPS
            
        Returns:
            Dictionary with sampling information
        """
        nframes = self.calculate_nframes(video_config, total_frames, video_fps)
        video_duration = total_frames / video_fps if video_fps > 0 else 0
        
        sampling_info = {
            "strategy": video_config.get("strategy", self.sampling_config.strategy),
            "total_frames": total_frames,
            "video_fps": video_fps,
            "video_duration": video_duration,
            "target_nframes": nframes,
            "sampling_fps": nframes / max(video_duration, 1e-6),
            "frame_factor": self.sampling_config.frame_factor,
            "min_frames": self.sampling_config.min_frames,
            "max_frames": self.sampling_config.max_frames,
            "compression_ratio": nframes / max(total_frames, 1),
        }
        
        # Add strategy-specific information
        if sampling_info["strategy"] == "duration":
            base_interval = video_config.get("base_interval", self.sampling_config.base_interval)
            sampling_info["base_interval"] = base_interval
            sampling_info["frames_per_interval"] = nframes / max(video_duration / base_interval, 1)
        
        return sampling_info
    
    def optimize_for_model(self, model_name: str, video_config: Dict[str, Any], 
                          total_frames: int, video_fps: float) -> int:
        """
        Optimize sampling for specific model requirements.
        
        Args:
            model_name: Name of the target model
            video_config: Video configuration
            total_frames: Total frames in video
            video_fps: Video FPS
            
        Returns:
            Optimized number of frames
        """
        # Model-specific optimizations
        if "qwen" in model_name.lower():
            # Qwen models prefer specific frame counts
            base_nframes = self._uniform_sampling(video_config, total_frames, video_fps)
            
            # Optimize for Qwen's attention patterns
            if base_nframes <= 16:
                optimized_nframes = 16
            elif base_nframes <= 32:
                optimized_nframes = 32
            elif base_nframes <= 64:
                optimized_nframes = 64
            else:
                optimized_nframes = min(base_nframes, 256)
            
            optimized_nframes = floor_by_factor(optimized_nframes, self.sampling_config.frame_factor)
            
        elif "llava" in model_name.lower():
            # LLaVA models might prefer different frame counts
            base_nframes = self._uniform_sampling(video_config, total_frames, video_fps)
            optimized_nframes = min(base_nframes, 128)  # LLaVA typical limit
            
        else:
            # Default optimization
            optimized_nframes = self._uniform_sampling(video_config, total_frames, video_fps)
        
        # Ensure within global constraints
        optimized_nframes = max(self.sampling_config.min_frames, optimized_nframes)
        optimized_nframes = min(self.sampling_config.max_frames, optimized_nframes)
        optimized_nframes = min(optimized_nframes, total_frames)
        
        logger.debug(
            f"Model optimization for {model_name}: "
            f"base_nframes -> optimized_nframes = {optimized_nframes}"
        )
        
        return optimized_nframes


class AdvancedSampler(SmartSampler):
    """
    Advanced sampler with additional features like scene detection and motion analysis.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._scene_cache = {}
    
    def analyze_video_content(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze video content for advanced sampling decisions.
        
        Args:
            video_tensor: Video tensor (T, C, H, W)
            
        Returns:
            Dictionary with content analysis results
        """
        if video_tensor.numel() == 0:
            return {"scene_changes": [], "motion_scores": [], "complexity": 0.0}
        
        # Simple scene change detection using frame differences
        scene_changes = self._detect_scene_changes(video_tensor)
        
        # Motion analysis
        motion_scores = self._calculate_motion_scores(video_tensor)
        
        # Overall complexity score
        complexity = self._calculate_complexity_score(video_tensor, scene_changes, motion_scores)
        
        return {
            "scene_changes": scene_changes,
            "motion_scores": motion_scores,
            "complexity": complexity,
            "num_scenes": len(scene_changes) + 1,
            "avg_motion": sum(motion_scores) / max(len(motion_scores), 1)
        }
    
    def _detect_scene_changes(self, video_tensor: torch.Tensor) -> List[int]:
        """Detect scene changes in video tensor."""
        if video_tensor.size(0) < 2:
            return []
        
        # Calculate frame differences
        frame_diffs = []
        for i in range(1, video_tensor.size(0)):
            diff = torch.mean((video_tensor[i] - video_tensor[i-1]) ** 2).item()
            frame_diffs.append(diff)
        
        # Find significant changes
        if not frame_diffs:
            return []
        
        threshold = self.sampling_config.scene_change_threshold
        mean_diff = sum(frame_diffs) / len(frame_diffs)
        adaptive_threshold = mean_diff * (1 + threshold)
        
        scene_changes = []
        for i, diff in enumerate(frame_diffs):
            if diff > adaptive_threshold:
                scene_changes.append(i + 1)  # Frame index after the change
        
        return scene_changes
    
    def _calculate_motion_scores(self, video_tensor: torch.Tensor) -> List[float]:
        """Calculate motion scores for each frame."""
        if video_tensor.size(0) < 2:
            return [0.0] * video_tensor.size(0)
        
        motion_scores = [0.0]  # First frame has no motion
        
        for i in range(1, video_tensor.size(0)):
            # Simple motion estimation using frame difference
            motion = torch.mean(torch.abs(video_tensor[i] - video_tensor[i-1])).item()
            motion_scores.append(motion)
        
        return motion_scores
    
    def _calculate_complexity_score(self, video_tensor: torch.Tensor, 
                                  scene_changes: List[int], motion_scores: List[float]) -> float:
        """Calculate overall video complexity score."""
        if video_tensor.numel() == 0:
            return 0.0
        
        # Factors contributing to complexity
        scene_complexity = len(scene_changes) / max(video_tensor.size(0), 1)
        motion_complexity = sum(motion_scores) / max(len(motion_scores), 1)
        
        # Combine factors
        complexity = (scene_complexity * 0.4 + motion_complexity * 0.6)
        
        return min(complexity, 1.0)  # Normalize to [0, 1]


@lru_cache(maxsize=1)
def get_default_sampler():
    """Get default sampler with standard configuration."""
    from ..config import get_default_config
    return SmartSampler(get_default_config())


def sample_frames_simple(total_frames: int, video_fps: float, target_fps: float = 2.0) -> int:
    """
    Simple function to calculate frame sampling.
    
    Args:
        total_frames: Total frames in video
        video_fps: Original video FPS
        target_fps: Target sampling FPS
        
    Returns:
        Number of frames to sample
    """
    sampler = get_default_sampler()
    video_config = {"fps": target_fps}
    return sampler.calculate_nframes(video_config, total_frames, video_fps) 