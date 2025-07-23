# Detailed Video Processing Pipeline Plan
*Based on Qwen2.5-VL Video Processing Architecture*

## Project Overview

Create a model-agnostic video processing pipeline that can efficiently process videos for any multimodal AI model, with configurable parameters and multiple output formats.

## 1. Architecture Overview

```
                            Main Processing Pipeline (Sequential)
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Frame Extractor │───▶│   Smart Sampler  │───▶│  Frame Processor│
└─────────────────┘    └──────────────────┘    └──────────────────┘    └─────────────────┘
                                                                                  │
                                                                                  ▼
                                                                         ┌─────────────────┐
                                                                         │   Tokenizer     │ (calculates tokens from dimensions)
                                                                         └─────────────────┘
                                                                                  │
                                                                                  ▼
                                                                         ┌─────────────────┐
                                                                         │  Format Handler │ (converts frames to output format)
                                                                         └─────────────────┘
                                                                                  │
                                                                                  ▼
                                                                         ┌──────────────────┐
                                                                         │  Output Manager  │
                                                                         └──────────────────┘

                            Supporting Systems (Cross-cutting)
                                                    ┌─────────────────┐
                                                    │   Config Mgmt   │ (provides config to all)
                                                    └─────────────────┘
                                                    ┌─────────────────┐
                                                    │   Cache System  │ (caches intermediate results)
                                                    └─────────────────┘
```

## 2. Component Specifications

### 2.1 Video Input Handler
**Purpose**: Accept various video input formats and normalize them
**Inputs**: 
- Local video files (MP4, MOV, AVI, etc.)
- URLs (HTTP/HTTPS)
- Base64 encoded videos
- Pre-extracted frame sequences
- Video byte streams

**Outputs**: Standardized video metadata and file handles

**Key Features**:
- Multi-format support with fallback mechanisms
- Video validation and metadata extraction
- Time range specification (start_time, end_time)
- Input sanitization and error handling

### 2.2 Frame Extractor
**Purpose**: Extract frames from video using multiple decoder backends with automatic fallback

#### 2.2.1 Backend Specifications

**TorchVision Backend**
- **Use Case**: Default backend for most video formats
- **Advantages**: 
  - Built into PyTorch ecosystem
  - Reliable CUDA support
  - Good format compatibility
- **Limitations**:
  - Slower than specialized decoders
  - Limited HTTP/HTTPS support in older versions (<0.19.0)
- **Configuration**:
  ```yaml
  torchvision:
    output_format: "TCHW"  # Tensor format
    pts_unit: "sec"        # Time unit for seeking
    backend: "pyav"        # Video backend
  ```
- **Frame Extraction Method**:
  - Uses `torchvision.io.read_video()`
  - Supports time-based range extraction (`start_pts`, `end_pts`)
  - Automatic FPS detection and metadata extraction

**Decord Backend**
- **Use Case**: High-performance video reading with multi-threading
- **Advantages**:
  - Fastest decoding performance
  - Excellent random access capabilities
  - Built-in keyframe indexing
  - Multi-threaded frame extraction
- **Limitations**:
  - Requires separate installation
  - Limited format support compared to FFmpeg
- **Configuration**:
  ```yaml
  decord:
    num_threads: 4         # Decoding threads
    ctx: "cpu"            # Device context
    width: -1             # Resize width (-1 = original)
    height: -1            # Resize height (-1 = original)
  ```
- **Frame Extraction Method**:
  - Uses `decord.VideoReader()`
  - Efficient batch frame extraction with `get_batch()`
  - Smart frame indexing with `np.linspace()` for uniform sampling

**TorchCodec Backend**
- **Use Case**: Maximum compatibility with FFmpeg-based decoding
- **Advantages**:
  - Best format compatibility (uses FFmpeg)
  - Handles corrupted or unusual video files
  - Precise frame seeking
  - Support for complex video containers
- **Limitations**:
  - Slower than Decord
  - Higher memory usage
  - Requires FFmpeg installation
- **Configuration**:
  ```yaml
  torchcodec:
    num_ffmpeg_threads: 8  # FFmpeg thread count
    device: "cpu"          # Processing device
    seek_mode: "approximate" # or "exact"
  ```
- **Frame Extraction Method**:
  - Uses `torchcodec.decoders.VideoDecoder()`
  - Precise frame extraction with `get_frames_at(indices=[])`
  - Advanced seeking capabilities

#### 2.2.2 Automatic Backend Selection & Fallback Chain

**Selection Priority** (configurable):
1. **TorchCodec** (most compatible)
2. **Decord** (fastest)
3. **TorchVision** (most reliable)

**Fallback Logic**:
```python
def get_video_reader_backend():
    backend_priority = ["torchcodec", "decord", "torchvision"]
    
    for backend in backend_priority:
        try:
            if backend == "torchcodec" and has_torchcodec():
                return backend
            elif backend == "decord" and has_decord():
                return backend
            elif backend == "torchvision":
                return backend
        except ImportError:
            continue
    
    raise RuntimeError("No video backend available")

def extract_frames_with_fallback(video_path, config):
    backend = get_video_reader_backend()
    
    for attempt_backend in [backend, "torchvision"]:  # Always fallback to torchvision
        try:
            return VIDEO_READER_BACKENDS[attempt_backend](video_path, config)
        except Exception as e:
            logger.warning(f"Backend {attempt_backend} failed: {e}")
            continue
    
    raise RuntimeError("All video backends failed")
```

#### 2.2.3 Common Interface Implementation

**Unified Frame Extraction Interface**:
```python
@dataclass
class VideoExtractionResult:
    frames: torch.Tensor          # Shape: (T, C, H, W)
    sample_fps: float            # Effective sampling rate
    metadata: Dict[str, Any]     # Video metadata
    backend_used: str           # Which backend was used

def extract_frames(video_config: Dict) -> VideoExtractionResult:
    """
    Universal frame extraction interface
    
    Args:
        video_config: {
            'video': str,                    # Video path/URL
            'fps': float,                    # Target FPS (optional)
            'nframes': int,                  # Target frame count (optional)
            'video_start': float,            # Start time in seconds (optional)
            'video_end': float,              # End time in seconds (optional)
            'min_frames': int,               # Minimum frames
            'max_frames': int,               # Maximum frames
        }
    
    Returns:
        VideoExtractionResult with standardized output
    """
```

**Backend-Specific Implementations**:

1. **TorchVision Implementation**:
   ```python
   def _read_video_torchvision(config):
       video, audio, info = torchvision.io.read_video(
           config['video'],
           start_pts=config.get('video_start', 0.0),
           end_pts=config.get('video_end', None),
           pts_unit="sec",
           output_format="TCHW"
       )
       
       total_frames, video_fps = video.size(0), info["video_fps"]
       nframes = calculate_smart_nframes(config, total_frames, video_fps)
       idx = torch.linspace(0, total_frames - 1, nframes).round().long()
       
       return video[idx], nframes / max(total_frames, 1e-6) * video_fps
   ```

2. **Decord Implementation**:
   ```python
   def _read_video_decord(config):
       vr = decord.VideoReader(config['video'], num_threads=4)
       total_frames, video_fps = len(vr), vr.get_avg_fps()
       
       start_frame, end_frame = calculate_frame_range(config, total_frames, video_fps)
       nframes = calculate_smart_nframes(config, total_frames, video_fps)
       
       idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
       video = vr.get_batch(idx).asnumpy()
       video = torch.tensor(video).permute(0, 3, 1, 2)  # NHWC -> NCHW
       
       return video, nframes / max(total_frames, 1e-6) * video_fps
   ```

3. **TorchCodec Implementation**:
   ```python
   def _read_video_torchcodec(config):
       decoder = torchcodec.decoders.VideoDecoder(
           config['video'], 
           num_ffmpeg_threads=8
       )
       
       total_frames = decoder.metadata.num_frames
       video_fps = decoder.metadata.average_fps
       
       start_frame, end_frame = calculate_frame_range(config, total_frames, video_fps)
       nframes = calculate_smart_nframes(config, total_frames, video_fps)
       
       idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
       video = decoder.get_frames_at(indices=idx).data
       
       return video, nframes / max(total_frames, 1e-6) * video_fps
   ```

#### 2.2.4 Performance Characteristics

**Benchmark Results** (approximate, depends on hardware):
| Backend    | Speed | Memory | Compatibility | Random Access |
|------------|-------|--------|---------------|---------------|
| TorchCodec | 🟡    | 🔴     | 🟢           | 🟢            |
| Decord     | 🟢    | 🟢     | 🟡           | 🟢            |
| TorchVision| 🟡    | 🟡     | 🟢           | 🟡            |

**Key Features**:
- Automatic backend selection with fallback chain
- Unified interface across all backends
- Configurable thread pools and memory management
- Error handling with detailed logging
- Performance monitoring and backend recommendation

### 2.3 Smart Sampler
**Purpose**: Intelligently sample frames based on video characteristics and model requirements

**Sampling Strategies**:
- **Uniform Sampling**: Even distribution across video timeline
- **Temporal Adaptive**: Dynamic sampling based on scene complexity
- **FPS-based**: Target specific frames per second output
- **Duration-based**: Sample based on video length intervals
- **Keyframe Priority**: Prefer keyframes when available

**Configuration Parameters**:
- `min_frames`: Minimum frames to extract (default: 4)
- `max_frames`: Maximum frames to extract (default: 768)
- `target_fps`: Desired output frame rate (default: 2.0)
- `frame_factor`: Frame alignment factor (default: 2)
- `base_interval`: Sampling interval in seconds (default: 4)

**Smart Frame Count Calculation**:
```
num_frames_to_sample = round(video_length / base_interval)
target_frames = min(max(num_frames_to_sample, min_frames), max_frames)
target_frames = round_to_nearest_factor(target_frames, frame_factor)
```

### 2.4 Frame Processor
**Purpose**: Resize and process extracted frames for model consumption

**Processing Pipeline**:
1. **Smart Resize Algorithm**: 
   - Maintain aspect ratio while meeting pixel constraints
   - Factor-based alignment (28x28 patches for Qwen2.5-VL)
   - Min/max pixel budget enforcement

2. **Pixel Budget Management**:
   - `min_pixels`: Minimum resolution per frame (default: 128×28×28)
   - `max_pixels`: Maximum resolution per frame (default: 768×28×28)
   - `total_pixels`: Total pixel budget across all frames
   - Dynamic max_pixels calculation: `max(min(VIDEO_MAX_PIXELS, total_pixels/nframes * FRAME_FACTOR), min_pixels * 1.05)`

3. **Image Enhancement**:
   - Bicubic interpolation for high-quality resizing
   - Anti-aliasing during resize operations
   - Color space normalization (RGB conversion)

### 2.5 Tokenizer
**Purpose**: Calculate token counts for model input planning

**Token Calculation Formula**:
```
tokens_per_frame = (resized_height / patch_size) * (resized_width / patch_size)
total_video_tokens = (num_frames / frame_factor) * tokens_per_frame
```

**Key Features**:
- Model-specific patch size configuration
- Memory estimation for different precisions (FP32, BF16, INT8, INT4)
- Token budget validation against model limits
- Batch size optimization recommendations

### 2.6 Format Handler
**Purpose**: Convert processed frames to model-specific formats

**Output Formats**:
- **PyTorch Tensors**: Shape (T, C, H, W) for direct model input
- **PIL Images**: List of PIL.Image objects for preprocessing pipelines
- **NumPy Arrays**: For custom processing workflows
- **Base64 Frames**: For API and web service integration
- **Video Metadata**: Frame timing, FPS, and grid information

**Format Configurations**:
- Tensor dtype selection (float32, float16, bfloat16)
- Channel ordering (RGB vs BGR)
- Normalization parameters (mean, std)
- Device placement (CPU, CUDA)

## 3. Configuration System

### 3.1 Model-Agnostic Parameters
```yaml
video_processing:
  # Input constraints
  max_video_duration: 3600  # seconds
  supported_formats: [mp4, mov, avi, mkv, webm]
  
  # Frame extraction
  decoder_priority: [torchcodec, decord, torchvision]
  num_threads: 8
  
  # Sampling parameters
  default_fps: 2.0
  min_frames: 4
  max_frames: 768
  frame_factor: 2
  base_interval: 4
  
  # Processing constraints
  image_factor: 28  # patch size
  min_pixels: 3584    # 128 * 28
  max_pixels: 602112  # 768 * 28 * 28
  total_pixels_ratio: 0.9  # of model's max context
  
  # Memory management
  cache_size: 100  # number of videos to cache
  batch_size: 32   # frames per processing batch
```

### 3.2 Model-Specific Configurations
```yaml
models:
  qwen2.5-vl:
    patch_size: 28
    max_context_tokens: 128000
    preferred_precision: bfloat16
    temporal_patch_size: 2
    
  llava:
    patch_size: 14
    max_context_tokens: 4096
    preferred_precision: float16
    
  gpt4v:
    max_frames: 64
    preferred_format: base64
    api_constraints:
      max_image_size: 20MB
```

## 4. Performance Optimization

### 4.1 Parallel Processing
- **Multi-threaded Frame Extraction**: Utilize all available CPU cores
- **Batch Processing**: Process multiple frames simultaneously
- **Async I/O**: Non-blocking file operations
- **GPU Acceleration**: CUDA-enabled resize operations when available

### 4.2 Caching Strategy
- **LRU Cache**: Recently processed videos stay in memory
- **Incremental Processing**: Cache intermediate results
- **Smart Invalidation**: Cache invalidation based on parameter changes
- **Disk Backup**: Persistent cache for expensive operations

### 4.3 Memory Management
- **Streaming Processing**: Process videos larger than available memory
- **Memory Pool**: Reuse tensor allocations
- **Garbage Collection**: Explicit cleanup of large objects
- **Memory Monitoring**: Track and limit memory usage

## 5. Output Specifications

### 5.1 Standard Output Format
```python
ProcessedVideo = {
    'frames': torch.Tensor,           # Shape: (N, C, H, W)
    'metadata': {
        'num_frames': int,
        'original_fps': float,
        'sample_fps': float,
        'duration': float,
        'resolution': tuple,          # (height, width)
        'total_tokens': int,
        'grid_thw': list,            # Temporal-Height-Width grid
    },
    'timing': {
        'extraction_time': float,
        'processing_time': float,
        'total_time': float,
    }
}
```

### 5.2 Alternative Formats
- **HuggingFace Format**: Compatible with transformers pipeline
- **OpenAI Format**: Base64 encoded for API consumption
- **Custom Format**: User-defined output structure
- **Streaming Format**: Iterator for large video processing

## 6. Error Handling & Robustness

### 6.1 Fallback Mechanisms
- **Decoder Fallback Chain**: Try multiple backends sequentially
- **Quality Degradation**: Reduce quality when memory/time constrained
- **Partial Processing**: Return partial results when possible
- **Graceful Failure**: Detailed error reporting with suggested fixes

### 6.2 Validation Pipeline
- **Input Validation**: File format, size, duration checks
- **Parameter Validation**: Ensure configuration consistency
- **Output Validation**: Verify token counts and dimensions
- **Quality Assurance**: Frame quality and completeness checks

## 7. Integration Points

### 7.1 Model Integration
- **Plugin Architecture**: Easy addition of new model support
- **Preprocessing Hooks**: Custom preprocessing for specific models
- **Postprocessing Hooks**: Custom output formatting
- **Model Discovery**: Automatic configuration detection

### 7.2 Framework Integration
- **PyTorch DataLoader**: Compatible dataset implementation
- **HuggingFace Datasets**: Integration with datasets library
- **MLflow**: Experiment tracking and model versioning
- **Weights & Biases**: Performance monitoring and optimization

## 8. Performance Metrics & Monitoring

### 8.1 Key Performance Indicators
- **Processing Speed**: Frames per second throughput
- **Memory Efficiency**: Peak memory usage per video
- **Quality Metrics**: Frame quality and sampling coverage
- **Error Rates**: Failure rates by video type and size

### 8.2 Profiling & Optimization
- **Bottleneck Identification**: Profile each pipeline component
- **Resource Utilization**: CPU, GPU, and memory usage tracking
- **Scalability Testing**: Performance under different loads
- **Optimization Recommendations**: Automatic parameter tuning

## 9. Testing Strategy

### 9.1 Unit Testing
- Component-level testing for each pipeline stage
- Edge case handling (corrupted videos, extreme resolutions)
- Parameter validation and boundary testing
- Memory leak detection

### 9.2 Integration Testing
- End-to-end pipeline testing with various video formats
- Model compatibility testing across different architectures
- Performance regression testing
- Cross-platform compatibility (Windows, Linux, macOS)

### 9.3 Benchmark Suite
- Standard video dataset processing
- Performance comparison with existing solutions
- Memory usage profiling under different configurations
- Accuracy validation against reference implementations

## 10. Future Enhancements

### 10.1 Advanced Features
- **Scene Detection**: Intelligent sampling based on scene changes
- **Content-Aware Sampling**: Focus on important visual content
- **Multi-Stream Support**: Handle videos with multiple streams
- **Real-time Processing**: Live video stream processing

### 10.2 Model-Specific Optimizations
- **Attention-Guided Sampling**: Use model attention to guide frame selection
- **Progressive Processing**: Start with low quality, refine as needed
- **Dynamic Resolution**: Adjust resolution based on content complexity
- **Temporal Coherence**: Maintain consistency across frame sequences
