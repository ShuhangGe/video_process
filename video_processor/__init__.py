"""
Video Processor - Model-agnostic video processing pipeline for multimodal AI models

A comprehensive video processing toolkit that efficiently processes videos for any multimodal
AI model, with configurable parameters and multiple output formats.

Main Features:
- Multi-backend video reading (TorchVision, Decord, TorchCodec)
- Smart frame sampling with configurable parameters
- Batch processing optimization
- Multiple output formats
"""

from .config import (
    VideoProcessorConfig, 
    get_default_config, 
    get_fast_config, 
    get_high_quality_config
)
from .core.video_input import VideoInputHandler
from .core.frame_extractor import FrameExtractor
from .core.smart_sampler import SmartSampler
from .core.frame_processor import FrameProcessor
from .core.tokenizer import VideoTokenizer
from .utils.format_handler import FormatHandler



__version__ = "0.1.0"
__all__ = [
    # Core classes
    "VideoProcessor",
    "VideoProcessorConfig", 
    "VideoInputHandler",
    "FrameExtractor",
    "SmartSampler", 
    "FrameProcessor",
    "VideoTokenizer",
    "FormatHandler",
    
    # Configuration functions
    "get_default_config",
    "get_fast_config", 
    "get_high_quality_config",
    
    # Main API functions
    "process_video",
    "process_videos_batch",
]


class VideoProcessor:
    """
    Main video processing pipeline that orchestrates all components.
    
    This is the primary interface for video processing, combining all components
    into a unified pipeline with configurable parameters and multiple output formats.
    """
    
    def __init__(self, config: VideoProcessorConfig = None):
        """Initialize the video processor with optional configuration."""
        self.config = config or VideoProcessorConfig()
        
        # Initialize core components
        self.input_handler = VideoInputHandler(self.config)
        self.frame_extractor = FrameExtractor(self.config)
        self.smart_sampler = SmartSampler(self.config)
        self.frame_processor = FrameProcessor(self.config)
        self.tokenizer = VideoTokenizer(self.config)
        self.format_handler = FormatHandler(self.config)
    
    def process(self, video_input, output_format=None, **kwargs):
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_input: Video file path, URL, or video configuration dict
            output_format: Output format ("standard", "huggingface", "openai", "streaming")
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedVideo object with frames, metadata, and timing information
        """
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        output_format = output_format or self.config.output.default_format
        
        # Start timing
        start_time = time.time()
        timing = {}
        
        try:
            # Step 1: Process input
            logger.debug(f"Processing video input: {str(video_input)[:100]}...")
            input_start = time.time()
            
            video_config = self.input_handler.process_input(video_input)
            if video_config.get("fallback", False):
                raise RuntimeError(f"Failed to process input: {video_config.get('error')}")
            
            # Merge additional kwargs into video config
            video_config.update(kwargs)
            timing["input_processing"] = time.time() - input_start
            

            
            # Step 2: Extract frames
            logger.debug("Extracting frames from video...")
            extraction_start = time.time()
            
            video_tensor, sample_fps = self.frame_extractor.extract_frames(video_config)
            timing["frame_extraction"] = time.time() - extraction_start
            
            if video_tensor is None or video_tensor.numel() == 0:
                raise RuntimeError("No frames extracted from video")
            
            logger.info(f"Extracted frames: {video_tensor.shape}, sample_fps: {sample_fps:.2f}")
            
            # Step 3: Process frames (resize, normalize, etc.)
            logger.debug("Processing frames...")
            processing_start = time.time()
            
            processed_tensor = self.frame_processor.process_frames(video_tensor, video_config)
            timing["frame_processing"] = time.time() - processing_start
            
            # Step 4: Calculate tokens and metadata
            logger.debug("Calculating tokens...")
            token_start = time.time()
            
            token_info = self.tokenizer.calculate_tokens(processed_tensor, video_config)
            timing["token_calculation"] = time.time() - token_start
            
            # Step 5: Compile metadata
            metadata = {
                "num_frames": processed_tensor.shape[0],
                "height": processed_tensor.shape[2],
                "width": processed_tensor.shape[3],
                "channels": processed_tensor.shape[1],
                "sample_fps": sample_fps,
                "video_fps": video_config.get("video_fps"),
                "input_type": video_config.get("input_type"),
                "processing_backend": getattr(self.frame_extractor, '_backend_priority', ['unknown'])[0],
                "grid_thw": token_info.get("grid_thw"),
            }
            
            # Add original video metadata if available
            if "original_path" in video_config:
                metadata["original_path"] = video_config["original_path"]
            if "file_size" in video_config:
                metadata["file_size"] = video_config["file_size"]
            
            # Step 6: Create ProcessedVideo result
            timing["total_time"] = time.time() - start_time
            
            # Quality assessment if configured
            quality_info = None
            if self.config.output.include_debug_info:
                quality_start = time.time()
                from .core.frame_processor import AdvancedFrameProcessor
                advanced_processor = AdvancedFrameProcessor(self.config)
                quality_info = advanced_processor.assess_frame_quality(processed_tensor)
                timing["quality_assessment"] = time.time() - quality_start
            
            # Processing information
            cache_info = {
                "cached": False,
                "cache_key": None
            }
            
            # Create ProcessedVideo object
            from .utils.format_handler import ProcessedVideo
            processed_video = ProcessedVideo(
                frames=processed_tensor,
                metadata=metadata,
                timing=timing,
                token_info=token_info,
                quality_info=quality_info,
                cache_info=cache_info
            )
            

            
            logger.info(
                f"Video processing completed: {processed_tensor.shape} in {timing['total_time']:.3f}s, "
                f"tokens: {token_info['total_tokens']}"
            )
            
            # Step 7: Format output
            format_start = time.time()
            formatted_result = self.format_handler.format_output(
                video_tensor=processed_tensor,
                metadata=metadata,
                timing=timing,
                format_type=output_format,
                token_info=token_info,
                quality_info=quality_info,
                cache_info=cache_info
            )
            timing["format_output"] = time.time() - format_start
            
            return formatted_result
            
        except Exception as e:
            timing["total_time"] = time.time() - start_time
            timing["error"] = str(e)
            
            logger.error(f"Video processing failed: {e}")
            
            if self.config.strict_mode:
                raise
            else:
                # Return error result in requested format
                error_metadata = {
                    "error": str(e),
                    "num_frames": 0,
                    "height": 0,
                    "width": 0,
                    "channels": 0,
                }
                
                return self.format_handler.format_output(
                    video_tensor=None,
                    metadata=error_metadata,
                    timing=timing,
                    format_type=output_format
                )
    
    def process_batch(self, video_inputs, output_format=None, **kwargs):
        """Process multiple videos in batch for better efficiency."""
        import logging
        import time
        
        logger = logging.getLogger(__name__)
        
        if not video_inputs:
            return []
        
        batch_start = time.time()
        output_format = output_format or self.config.output.default_format
        
        # Sequential processing
        logger.info(f"Processing batch of {len(video_inputs)} videos sequentially")
        results = []
        
        for i, video_input in enumerate(video_inputs):
            try:
                logger.debug(f"Processing video {i+1}/{len(video_inputs)}")
                result = self.process(video_input, output_format=output_format, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process video {i+1}: {e}")
                if self.config.strict_mode:
                    raise
                else:
                    # Add error result
                    error_result = self.format_handler.format_output(
                        video_tensor=None,
                        metadata={"error": str(e), "batch_index": i},
                        timing={"error": True},
                        format_type=output_format
                    )
                    results.append(error_result)
        
        batch_time = time.time() - batch_start
        logger.info(f"Batch processing completed: {len(results)} videos in {batch_time:.3f}s")
        
        return results
    

    def get_system_info(self):
        """Get comprehensive system information."""
        return {
            "config": {
                "strict_mode": self.config.strict_mode,
                "device": self.config.device,
                "dtype": self.config.dtype,
            },
            "backends": self.frame_extractor.get_backend_info(),
            "format_info": self.format_handler.get_format_info(),
            "tokenizer_info": self.tokenizer.get_tokenizer_info(),
        }
    
    def benchmark(self, video_input, num_runs=3):
        """Benchmark processing performance."""
        import time
        
        times = []
        results = []
        
        for run in range(num_runs):
            start_time = time.time()
            result = self.process(video_input, output_format="raw")
            end_time = time.time()
            
            times.append(end_time - start_time)
            if run == 0:  # Keep first result for analysis
                results.append(result)
        
        return {
            "runs": num_runs,
            "times": times,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "first_result": results[0] if results else None
        }


def process_video(video_input, config=None, **kwargs):
    """
    Convenience function to process a single video.
    
    Args:
        video_input: Video file path, URL, or configuration dict
        config: Optional VideoProcessorConfig
        **kwargs: Additional processing parameters
        
    Returns:
        ProcessedVideo object
    """
    processor = VideoProcessor(config)
    return processor.process(video_input, **kwargs)


def process_videos_batch(video_inputs, config=None, **kwargs):
    """
    Convenience function to process multiple videos in batch.
    
    Args:
        video_inputs: List of video inputs
        config: Optional VideoProcessorConfig  
        **kwargs: Additional processing parameters
        
    Returns:
        List of ProcessedVideo objects
    """
    processor = VideoProcessor(config)
    return processor.process_batch(video_inputs, **kwargs)


 