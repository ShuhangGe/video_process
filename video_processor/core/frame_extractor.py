"""
Frame Extractor - Multi-backend video frame extraction with automatic fallback.

This module provides robust video frame extraction using multiple backends with
automatic fallback functionality. Based on the Qwen2.5-VL implementation.
"""

import time
import logging
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import torch
import torchvision
from packaging import version


logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Multi-backend frame extractor with automatic fallback chain.
    
    Supports three backends in order of preference:
    1. TorchCodec (most compatible)
    2. Decord (fastest)  
    3. TorchVision (most reliable)
    """
    
    def __init__(self, config):
        """Initialize frame extractor with configuration."""
        self.config = config
        self.backend_config = config.backend
        self._available_backends = self._detect_available_backends()
        self._backend_priority = self._determine_backend_priority()
        
        logger.info(f"Available backends: {self._available_backends}")
        logger.info(f"Backend priority: {self._backend_priority}")
    
    def extract_frames(self, video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """
        Extract frames from video using the best available backend.
        
        Args:
            video_config: Video configuration dictionary with 'video' path
            
        Returns:
            Tuple of (video_tensor, sample_fps)
            - video_tensor: Shape (T, C, H, W) 
            - sample_fps: Effective FPS of extracted frames
        """
        if video_config.get("fallback", False):
            raise RuntimeError(f"Cannot extract frames from fallback config: {video_config.get('error')}")
        
        video_path = video_config["video"]
        if isinstance(video_path, list):
            # Handle pre-extracted frames
            return self._process_frame_list(video_path, video_config)
        
        # Try backends in priority order
        last_error = None
        
        for backend_name in self._backend_priority:
            if backend_name not in self._available_backends:
                continue
                
            try:
                logger.debug(f"Attempting video extraction with {backend_name}")
                start_time = time.time()
                
                video_tensor, sample_fps = self._extract_with_backend(
                    backend_name, video_config
                )
                
                extraction_time = time.time() - start_time
                logger.info(
                    f"Successfully extracted frames using {backend_name} "
                    f"in {extraction_time:.3f}s, shape: {video_tensor.shape}, fps: {sample_fps:.2f}"
                )
                
                return video_tensor, sample_fps
                
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                last_error = e
                if self.config.strict_mode:
                    raise
                continue
        
        # All backends failed
        error_msg = f"All backends failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _extract_with_backend(self, backend_name: str, video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Extract frames using specific backend."""
        if backend_name == "torchvision":
            return self._extract_torchvision(video_config)
        elif backend_name == "decord":
            return self._extract_decord(video_config)
        elif backend_name == "torchcodec":
            return self._extract_torchcodec(video_config)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def _extract_torchvision(self, video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Extract frames using TorchVision backend."""
        video_path = video_config["video"]
        config = self.backend_config.torchvision
        
        # Check for HTTP/HTTPS support in older versions
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        
        st = time.time()
        video, audio, info = torchvision.io.read_video(
            video_path,
            start_pts=video_config.get("video_start", 0.0),
            end_pts=video_config.get("video_end", None),
            pts_unit=config["pts_unit"],
            output_format=config["output_format"],
        )
        
        total_frames = video.size(0)
        video_fps = info.get("video_fps", 30.0)  # Default to 30 fps if not available
        
        logger.debug(f"torchvision: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        
        # Handle empty video case
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            # Return a single black frame as fallback
            video = torch.zeros(1, 3, 224, 224)
            return video, video_fps
        
        nframes = self._calculate_target_frames(video_config, total_frames, video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video = video[idx]
        
        return video, sample_fps
    
    def _extract_decord(self, video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Extract frames using Decord backend."""
        import decord
        
        video_path = video_config["video"]
        config = self.backend_config.decord
        
        st = time.time()
        vr = decord.VideoReader(
            video_path,
            ctx=decord.cpu(0) if config["ctx"] == "cpu" else decord.gpu(0),
            num_threads=config["num_threads"]
        )
        
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        
        # Calculate frame range based on time constraints
        start_frame, end_frame, actual_total_frames = self._calculate_video_frame_range(
            video_config, total_frames, video_fps
        )
        
        nframes = self._calculate_target_frames(video_config, actual_total_frames, video_fps)
        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        logger.debug(f"decord: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        
        sample_fps = nframes / max(actual_total_frames, 1e-6) * video_fps
        return video, sample_fps
    
    def _extract_torchcodec(self, video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Extract frames using TorchCodec backend."""
        from torchcodec.decoders import VideoDecoder
        
        video_path = video_config["video"]
        config = self.backend_config.torchcodec
        
        st = time.time()
        decoder = VideoDecoder(
            video_path, 
            num_ffmpeg_threads=config["num_ffmpeg_threads"]
        )
        
        video_fps = decoder.metadata.average_fps
        total_frames = decoder.metadata.num_frames
        
        # Calculate frame range based on time constraints
        start_frame, end_frame, actual_total_frames = self._calculate_video_frame_range(
            video_config, total_frames, video_fps
        )
        
        nframes = self._calculate_target_frames(video_config, actual_total_frames, video_fps)
        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        
        sample_fps = nframes / max(actual_total_frames, 1e-6) * video_fps
        video = decoder.get_frames_at(indices=idx).data
        
        logger.debug(f"torchcodec: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        
        return video, sample_fps
    
    def _process_frame_list(self, frame_paths: List[str], video_config: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Process pre-extracted frame list."""
        from PIL import Image
        import torchvision.transforms as transforms
        
        if not frame_paths:
            raise ValueError("Frame list cannot be empty")
        
        # Load and process frames
        frames = []
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path).convert('RGB')
                # Convert to tensor (C, H, W)
                tensor = transforms.ToTensor()(image)
                frames.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_path}: {e}")
                if self.config.strict_mode:
                    raise
        
        if not frames:
            raise RuntimeError("No valid frames found in frame list")
        
        # Stack frames to create video tensor (T, C, H, W)
        video_tensor = torch.stack(frames, dim=0)
        
        # Estimate FPS (default to config target_fps)
        sample_fps = video_config.get("fps", self.config.sampling.target_fps)
        
        logger.info(f"Processed frame list: {len(frames)} frames, shape: {video_tensor.shape}")
        
        return video_tensor, sample_fps
    
    def _calculate_video_frame_range(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> Tuple[int, int, int]:
        """Calculate start and end frame indices based on time range."""
        video_start = video_config.get("video_start", None)
        video_end = video_config.get("video_end", None)
        
        if video_start is None and video_end is None:
            return 0, total_frames - 1, total_frames
        
        max_duration = total_frames / video_fps
        
        # Process start frame
        if video_start is not None:
            video_start_clamped = max(0.0, min(video_start, max_duration))
            start_frame = int(video_start_clamped * video_fps)
        else:
            start_frame = 0
        
        # Process end frame
        if video_end is not None:
            video_end_clamped = max(0.0, min(video_end, max_duration))
            end_frame = int(video_end_clamped * video_fps)
            end_frame = min(end_frame, total_frames - 1)
        else:
            end_frame = total_frames - 1
        
        # Validate frame order
        if start_frame >= end_frame:
            raise ValueError(
                f"Invalid time range: Start frame {start_frame} (at {video_start}s) "
                f"exceeds end frame {end_frame} (at {video_end}s). "
                f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
            )
        
        actual_total_frames = end_frame - start_frame + 1
        
        logger.debug(
            f"Frame range: {start_frame}-{end_frame} ({actual_total_frames} frames) "
            f"from {video_start}s-{video_end}s @ {video_fps}fps"
        )
        
        return start_frame, end_frame, actual_total_frames
    
    def _calculate_target_frames(self, video_config: Dict[str, Any], total_frames: int, video_fps: float) -> int:
        """Calculate the number of frames to extract."""
        from ..core.smart_sampler import SmartSampler
        
        # Use the smart sampler to determine target frame count
        sampler = SmartSampler(self.config)
        return sampler.calculate_nframes(video_config, total_frames, video_fps)
    
    def _detect_available_backends(self) -> List[str]:
        """Detect which backends are available."""
        available = []
        
        # Check TorchVision (always available)
        available.append("torchvision")
        
        # Check Decord
        try:
            import decord
            available.append("decord")
        except ImportError:
            logger.debug("Decord not available")
        
        # Check TorchCodec
        try:
            import torchcodec
            from torchcodec.decoders import VideoDecoder
            available.append("torchcodec")
        except ImportError:
            logger.debug("TorchCodec not available")
        
        return available
    
    def _determine_backend_priority(self) -> List[str]:
        """Determine backend priority based on configuration and availability."""
        if self.backend_config.force_backend:
            # Force specific backend if configured
            forced = self.backend_config.force_backend
            if forced in self._available_backends:
                return [forced]
            else:
                logger.warning(f"Forced backend {forced} not available, using auto-selection")
        
        # Use configured priority, filtered by availability
        priority = []
        for backend in self.backend_config.priority:
            if backend in self._available_backends:
                priority.append(backend)
        
        # Add any available backends not in the priority list
        for backend in self._available_backends:
            if backend not in priority:
                priority.append(backend)
        
        return priority
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        info = {
            "available_backends": self._available_backends,
            "backend_priority": self._backend_priority,
            "forced_backend": self.backend_config.force_backend
        }
        
        # Add version information
        versions = {}
        
        # TorchVision version
        try:
            versions["torchvision"] = torchvision.__version__
        except:
            versions["torchvision"] = "unknown"
        
        # Decord version
        try:
            import decord
            versions["decord"] = getattr(decord, "__version__", "unknown")
        except ImportError:
            versions["decord"] = "not_available"
        
        # TorchCodec version  
        try:
            import torchcodec
            versions["torchcodec"] = getattr(torchcodec, "__version__", "unknown")
        except ImportError:
            versions["torchcodec"] = "not_available"
        
        info["versions"] = versions
        return info
    
    def benchmark_backends(self, video_path: str, num_runs: int = 3) -> Dict[str, Dict[str, float]]:
        """Benchmark all available backends on a video file."""
        results = {}
        
        base_config = {"video": video_path}
        
        for backend_name in self._available_backends:
            times = []
            shapes = []
            fps_values = []
            
            for run in range(num_runs):
                try:
                    start_time = time.time()
                    video_tensor, sample_fps = self._extract_with_backend(backend_name, base_config)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    shapes.append(video_tensor.shape)
                    fps_values.append(sample_fps)
                    
                except Exception as e:
                    logger.warning(f"Benchmark failed for {backend_name} run {run}: {e}")
                    continue
            
            if times:
                results[backend_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "avg_fps": sum(fps_values) / len(fps_values),
                    "shape": shapes[0] if shapes else None,
                    "successful_runs": len(times),
                    "total_runs": num_runs
                }
            else:
                results[backend_name] = {
                    "error": "All runs failed",
                    "successful_runs": 0,
                    "total_runs": num_runs
                }
        
        return results


@lru_cache(maxsize=1)
def get_default_frame_extractor():
    """Get default frame extractor with standard configuration."""
    from ..config import get_default_config
    return FrameExtractor(get_default_config())


def extract_frames_simple(video_path: str, **kwargs) -> Tuple[torch.Tensor, float]:
    """
    Simple function to extract frames from a video.
    
    Args:
        video_path: Path to video file
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (video_tensor, sample_fps)
    """
    extractor = get_default_frame_extractor()
    video_config = {"video": video_path, **kwargs}
    return extractor.extract_frames(video_config) 