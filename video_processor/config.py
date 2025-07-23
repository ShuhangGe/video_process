"""
Configuration management system for the video processor.

This module provides a comprehensive configuration system that allows users to customize
all aspects of the video processing pipeline, from input handling to VLLM integration.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


logger = logging.getLogger(__name__)

# Default constants (can be overridden by config)
DEFAULT_IMAGE_FACTOR = 28
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28
DEFAULT_MAX_RATIO = 200

DEFAULT_VIDEO_MIN_PIXELS = 128 * 28 * 28  
DEFAULT_VIDEO_MAX_PIXELS = 768 * 28 * 28
DEFAULT_FRAME_FACTOR = 2
DEFAULT_FPS = 2.0
DEFAULT_FPS_MIN_FRAMES = 4
DEFAULT_FPS_MAX_FRAMES = 768

# VLLM-aware video total pixels (90% of max tokens for safety)
DEFAULT_VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))


@dataclass
class BackendConfig:
    """Configuration for video reading backends."""
    
    # Backend selection priority
    priority: List[str] = field(default_factory=lambda: ["torchcodec", "decord", "torchvision"])
    
    # Force specific backend (overrides auto-selection)
    force_backend: Optional[str] = None
    
    # TorchVision configuration
    torchvision: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "TCHW",
        "pts_unit": "sec",
        "backend": "pyav"
    })
    
    # Decord configuration  
    decord: Dict[str, Any] = field(default_factory=lambda: {
        "num_threads": 4,
        "ctx": "cpu",
        "width": -1,
        "height": -1
    })
    
    # TorchCodec configuration
    torchcodec: Dict[str, Any] = field(default_factory=lambda: {
        "num_ffmpeg_threads": int(os.environ.get('TORCHCODEC_NUM_THREADS', 8)),
        "device": "cpu", 
        "seek_mode": "approximate"
    })


@dataclass 
class SamplingConfig:
    """Configuration for smart frame sampling."""
    
    # Frame count parameters
    min_frames: int = DEFAULT_FPS_MIN_FRAMES
    max_frames: int = DEFAULT_FPS_MAX_FRAMES
    frame_factor: int = DEFAULT_FRAME_FACTOR
    
    # FPS-based sampling
    target_fps: float = DEFAULT_FPS
    base_interval: float = 4.0  # Sampling interval in seconds
    
    # Sampling strategies
    strategy: str = "uniform"  # "uniform", "adaptive", "keyframe", "duration"
    
    # Adaptive sampling parameters
    scene_change_threshold: float = 0.3
    motion_threshold: float = 0.1
    
    # Duration-based parameters
    short_video_threshold: float = 10.0  # seconds
    long_video_threshold: float = 300.0  # seconds


@dataclass
class ProcessingConfig:
    """Configuration for frame processing and resizing."""
    
    # Image processing parameters
    image_factor: int = DEFAULT_IMAGE_FACTOR
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_aspect_ratio: float = DEFAULT_MAX_RATIO
    
    # Video processing parameters
    video_min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS
    video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    video_total_pixels: int = DEFAULT_VIDEO_TOTAL_PIXELS
    
    # Processing options
    interpolation_mode: str = "bicubic"
    antialias: bool = True
    normalize: bool = False
    
    # Memory management
    max_batch_size: int = 8
    enable_half_precision: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None  # Default: ~/.cache/video_processor
    max_cache_size: int = 10 * 1024 * 1024 * 1024  # 10GB default
    
    # Cache strategies
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    memory_cache_size: int = 1024  # Number of items
    
    # Cache invalidation
    cache_ttl: int = 7 * 24 * 3600  # 7 days default
    invalidate_on_config_change: bool = True
    
    # Performance tuning
    async_cache_writes: bool = True
    compression_level: int = 6  # 0-9, higher = more compression


@dataclass
class VLLMConfig:
    """Configuration for VLLM integration."""
    
    # VLLM settings
    enable_vllm: bool = False
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_model_len: int = 5632
    max_num_seqs: int = 5
    
    # Multi-modal settings
    limit_mm_per_prompt: Dict[str, int] = field(default_factory=lambda: {
        "video": 1, 
        "image": 5,
        "audio": 1
    })
    
    # Performance settings
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", etc.
    
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 256
    top_p: float = 0.95
    top_k: int = 50
    
    # Serving configuration
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: Optional[str] = None
    
    # API compatibility
    openai_compatible: bool = True
    enable_streaming: bool = True


@dataclass
class OutputConfig:
    """Configuration for output formats and handling."""
    
    # Default output format
    default_format: str = "standard"  # "standard", "huggingface", "openai", "streaming"
    
    # Format-specific settings
    include_timing: bool = True
    include_metadata: bool = True
    include_debug_info: bool = False
    
    # Standard format options
    standard_format: Dict[str, Any] = field(default_factory=lambda: {
        "tensor_dtype": "float32",
        "include_fps": True,
        "include_grid_thw": True
    })
    
    # HuggingFace format options
    huggingface_format: Dict[str, Any] = field(default_factory=lambda: {
        "processor_class": "Qwen2_5_VLProcessor",
        "return_tensors": "pt"
    })
    
    # OpenAI format options
    openai_format: Dict[str, Any] = field(default_factory=lambda: {
        "base64_encode": True,
        "include_usage": True
    })
    
    # Streaming format options
    streaming_format: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 4,
        "overlap": 1
    })


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Performance monitoring
    enable_profiling: bool = False
    enable_timing: bool = True
    enable_memory_tracking: bool = False
    
    # Debug settings
    debug_mode: bool = False
    save_intermediate_results: bool = False
    debug_output_dir: Optional[str] = None


@dataclass
class VideoProcessorConfig:
    """
    Main configuration class that combines all component configurations.
    
    This class provides a comprehensive configuration system for the entire video
    processing pipeline, allowing fine-tuned control over all aspects of processing.
    """
    
    # Component configurations
    backend: BackendConfig = field(default_factory=BackendConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig) 
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    num_workers: int = 4
    
    # Error handling
    strict_mode: bool = False  # If True, raise errors instead of warnings
    fallback_enabled: bool = True
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        self._setup_logging()
        self._setup_cache_dir()
        self._validate_config()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.logging.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format=self.logging.log_format,
            filename=self.logging.log_file
        )
        
        if self.logging.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def _setup_cache_dir(self):
        """Setup cache directory if not specified."""
        if self.cache.cache_dir is None:
            cache_dir = Path.home() / ".cache" / "video_processor"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache.cache_dir = str(cache_dir)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate sampling parameters
        if self.sampling.min_frames >= self.sampling.max_frames:
            raise ValueError("min_frames must be less than max_frames")
        
        if self.sampling.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        
        # Validate processing parameters
        if self.processing.min_pixels >= self.processing.max_pixels:
            raise ValueError("min_pixels must be less than max_pixels")
        
        # Validate VLLM parameters
        if self.vllm.enable_vllm:
            if self.vllm.max_model_len <= 0:
                raise ValueError("max_model_len must be positive")
            
            if not (0 < self.vllm.gpu_memory_utilization <= 1):
                raise ValueError("gpu_memory_utilization must be between 0 and 1")
        
        logger.info("Configuration validation passed")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VideoProcessorConfig':
        """Create configuration from dictionary."""
        # Convert nested dictionaries to dataclass instances
        def convert_section(section_dict, section_class):
            if isinstance(section_dict, dict):
                return section_class(**section_dict)
            return section_dict
        
        config_dict = config_dict.copy()
        
        # Convert component configurations
        if 'backend' in config_dict:
            config_dict['backend'] = convert_section(config_dict['backend'], BackendConfig)
        if 'sampling' in config_dict:
            config_dict['sampling'] = convert_section(config_dict['sampling'], SamplingConfig)
        if 'processing' in config_dict:
            config_dict['processing'] = convert_section(config_dict['processing'], ProcessingConfig)
        if 'cache' in config_dict:
            config_dict['cache'] = convert_section(config_dict['cache'], CacheConfig)
        if 'vllm' in config_dict:
            config_dict['vllm'] = convert_section(config_dict['vllm'], VLLMConfig)
        if 'output' in config_dict:
            config_dict['output'] = convert_section(config_dict['output'], OutputConfig)
        if 'logging' in config_dict:
            config_dict['logging'] = convert_section(config_dict['logging'], LoggingConfig)
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'VideoProcessorConfig':
        """Load configuration from file (JSON or YAML)."""
        import json
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, config_path: Union[str, Path], format: str = "auto"):
        """Save configuration to file."""
        import json
        
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Determine format
        if format == "auto":
            format = "yaml" if config_path.suffix.lower() in ['.yaml', '.yml'] else "json"
        
        with open(config_path, 'w') as f:
            if format == "yaml":
                try:
                    import yaml
                    yaml.dump(config_dict, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                json.dump(config_dict, f, indent=2)
    
    def enable_vllm_integration(self, **vllm_kwargs):
        """Enable VLLM integration with optional parameters."""
        self.vllm.enable_vllm = True
        for key, value in vllm_kwargs.items():
            if hasattr(self.vllm, key):
                setattr(self.vllm, key, value)
    
    def disable_vllm_integration(self):
        """Disable VLLM integration."""
        self.vllm.enable_vllm = False
    
    def enable_caching(self, **cache_kwargs):
        """Enable caching with optional parameters."""
        self.cache.enable_cache = True
        for key, value in cache_kwargs.items():
            if hasattr(self.cache, key):
                setattr(self.cache, key, value)
    
    def disable_caching(self):
        """Disable caching."""
        self.cache.enable_cache = False
    
    def set_device(self, device: str):
        """Set processing device."""
        self.device = device
        if device.startswith("cuda") and self.vllm.enable_vllm:
            self.vllm.tensor_parallel_size = 1  # Adjust if needed
    
    def optimize_for_memory(self):
        """Apply memory optimization settings."""
        self.processing.enable_half_precision = True
        self.processing.max_batch_size = 4
        self.cache.memory_cache_size = 512
        self.vllm.gpu_memory_utilization = 0.7
    
    def optimize_for_speed(self):
        """Apply speed optimization settings.""" 
        self.backend.priority = ["decord", "torchcodec", "torchvision"]
        self.processing.max_batch_size = 16
        self.cache.async_cache_writes = True
        self.vllm.gpu_memory_utilization = 0.9


# Predefined configurations for common use cases
def get_default_config() -> VideoProcessorConfig:
    """Get default configuration for general use."""
    return VideoProcessorConfig()


def get_high_quality_config() -> VideoProcessorConfig:
    """Get configuration optimized for high quality processing."""
    config = VideoProcessorConfig()
    config.processing.max_pixels = 32 * 28 * 28 * 28  # Higher resolution
    config.sampling.max_frames = 1024  # More frames
    config.processing.interpolation_mode = "bicubic"
    config.processing.antialias = True
    return config


def get_fast_config() -> VideoProcessorConfig:
    """Get configuration optimized for speed."""
    config = VideoProcessorConfig()
    config.optimize_for_speed()
    config.processing.max_pixels = 4 * 28 * 28  # Lower resolution
    config.sampling.max_frames = 256  # Fewer frames
    return config


def get_vllm_config(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> VideoProcessorConfig:
    """Get configuration with VLLM integration enabled."""
    config = VideoProcessorConfig()
    config.enable_vllm_integration(model_name=model_name)
    return config 