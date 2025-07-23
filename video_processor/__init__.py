"""
Video Processor - Model-agnostic video processing pipeline for multimodal AI models

A comprehensive video processing toolkit that efficiently processes videos for any multimodal
AI model, with configurable parameters, multiple output formats, and VLLM integration.

Main Features:
- Multi-backend video reading (TorchVision, Decord, TorchCodec)
- Smart frame sampling with configurable parameters
- VLLM integration for efficient inference
- Batch processing optimization
- Caching system for performance
- OpenAI-compatible API server
- Multiple output formats
"""

from .config import VideoProcessorConfig
from .core.video_input import VideoInputHandler
from .core.frame_extractor import FrameExtractor
from .core.smart_sampler import SmartSampler
from .core.frame_processor import FrameProcessor
from .core.tokenizer import VideoTokenizer
from .utils.format_handler import FormatHandler
from .utils.cache_system import CacheSystem

try:
    from .vllm.vllm_integration import VLLMVideoProcessor
    from .vllm.batch_processor import BatchProcessor
    from .vllm.serving import VideoAPIServer
    VLLM_AVAILABLE = True
except ImportError:
    VLLMVideoProcessor = None
    BatchProcessor = None
    VideoAPIServer = None
    VLLM_AVAILABLE = False

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
    "CacheSystem",
    
    # VLLM components (if available)
    "VLLMVideoProcessor",
    "BatchProcessor", 
    "VideoAPIServer",
    
    # Main API functions
    "process_video",
    "process_videos_batch",
    "create_api_server",
    
    # Constants
    "VLLM_AVAILABLE",
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
        self.cache_system = CacheSystem(self.config)
        
        # Initialize VLLM components if available
        if VLLM_AVAILABLE and self.config.enable_vllm:
            self.vllm_processor = VLLMVideoProcessor(self.config)
            self.batch_processor = BatchProcessor(self.config)
        else:
            self.vllm_processor = None
            self.batch_processor = None
    
    def process(self, video_input, output_format="standard", **kwargs):
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_input: Video file path, URL, or video configuration dict
            output_format: Output format ("standard", "huggingface", "openai", "streaming")
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedVideo object with frames, metadata, and timing information
        """
        # Implementation will be in the actual class
        pass
    
    def process_batch(self, video_inputs, **kwargs):
        """Process multiple videos in batch for better efficiency."""
        if self.batch_processor:
            return self.batch_processor.process_batch(video_inputs, **kwargs)
        else:
            # Fallback to sequential processing
            return [self.process(video, **kwargs) for video in video_inputs]
    
    def create_server(self, **server_config):
        """Create an API server for video processing."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("VLLM integration required for API server")
        return VideoAPIServer(self.config, **server_config)


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


def create_api_server(config=None, **server_config):
    """
    Convenience function to create an API server.
    
    Args:
        config: Optional VideoProcessorConfig
        **server_config: Server configuration parameters
        
    Returns:
        VideoAPIServer instance
    """
    processor = VideoProcessor(config)
    return processor.create_server(**server_config) 