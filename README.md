# Video Processing Pipeline

A comprehensive video processing toolkit that combines multiple video processing capabilities including model-agnostic video processing for multimodal AI models and specialized TikTok video downloading with audio transcription.

## 🏗️ Project Components

### 1. 🎬 **Video Processor** (`video_processor/`)
A model-agnostic video processing pipeline designed for multimodal AI models with configurable parameters and multiple output formats.

**Key Features:**
- **🔧 Multi-backend Video Reading**: Supports TorchVision, Decord, and TorchCodec with automatic fallback
- **🎯 Smart Frame Sampling**: Intelligent sampling strategies (uniform, adaptive, keyframe, duration-based)
- **📏 Smart Resizing**: Maintains aspect ratios while optimizing for model requirements
- **🧮 Token Calculation**: Estimates token counts and memory requirements for different models
- **📤 Multiple Output Formats**: Standard, HuggingFace, OpenAI, Streaming, and Raw formats
- **⚡ Performance Optimized**: Batch processing, memory management, and async operations

### 2. 📱 **TikTok Video Downloader with Transcription** (`tt_video_download/`)
Enhanced TikTok video downloader that extracts audio and provides automatic speech-to-text transcription with speaker identification.

**Key Features:**
- ✅ **Video Downloading**: Download TikTok videos from URLs with metadata
- 🎵 **Audio Extraction**: Extract audio from videos using FFmpeg
- 🎤 **Speech Recognition**: Convert speech to text using Whisper Large-v3 Turbo
- 👥 **Speaker Diarization**: Identify different speakers using Pyannote
- ⏰ **Timestamp Alignment**: Precise timing for each speech segment
- 📊 **Enhanced Metadata**: Rich JSON output with transcription data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- CUDA compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd video_process
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Or using conda
conda install -c conda-forge ffmpeg
```

## 📖 Usage

### Video Processor

```python
from video_processor import VideoProcessor, get_default_config

# Initialize with default configuration
config = get_default_config()
processor = VideoProcessor(config)

# Process a single video
result = processor.process_video("path/to/video.mp4")

# Process multiple videos
results = processor.process_videos_batch([
    "video1.mp4", 
    "video2.mp4"
])
```

**Configuration Options:**
```python
from video_processor import get_fast_config, get_high_quality_config

# For fast processing
fast_config = get_fast_config()

# For high quality output
hq_config = get_high_quality_config()

# Custom configuration
config = VideoProcessorConfig(
    target_fps=2.0,
    max_frames=64,
    target_size=(224, 224),
    backend_preference=['torchvision', 'decord']
)
```

### TikTok Video Downloader

```bash
cd tt_video_download

# Setup (first time only)
python setup_transcription.py

# Create URL file
echo "https://www.tiktok.com/@username/video/1234567890" > urls.txt

# Run downloader with transcription
conda activate python12  # or python11
python get_tt.py
```

## 📁 Project Structure

```
video_process/
├── video_processor/          # Main video processing pipeline
│   ├── __init__.py           # Main API
│   ├── config.py             # Configuration management
│   ├── core/                 # Core processing components
│   ├── backends/             # Video reading backends
│   ├── utils/                # Utility functions
│   └── examples/             # Usage examples
├── tt_video_download/        # TikTok downloader with transcription
│   ├── get_tt.py             # Main downloader script
│   ├── setup_transcription.py # Setup script
│   ├── requirements.txt      # Specific dependencies
│   └── README.md             # Detailed documentation
├── read_video.py             # Core video reading functionality
├── video_process_plan.md     # Technical architecture document
├── requirements.txt          # Main project dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Video Processor Configuration

The video processor supports multiple configuration presets:

- **Default**: Balanced performance and quality
- **Fast**: Optimized for speed
- **High Quality**: Optimized for output quality

Custom configurations can be created by modifying:
- Frame sampling parameters
- Resizing strategies
- Backend preferences
- Token calculation settings

### TikTok Downloader Configuration

Configure the TikTok downloader in `tt_video_download/get_tt.py`:

```python
# Basic settings
urls_file = "urls.txt"
output_directory = "/path/to/output"
enable_transcription = True  # Set to False to skip audio processing

# Advanced settings (in the script)
whisper_model = "openai/whisper-large-v3-turbo"
pyannote_model = "pyannote/speaker-diarization-3.1"
```

## 📊 Output Formats

### Video Processor Outputs

- **Standard**: Basic tensor format
- **HuggingFace**: Compatible with HuggingFace models
- **OpenAI**: Compatible with OpenAI API format
- **Streaming**: For real-time processing
- **Raw**: Unprocessed frames

### TikTok Downloader Outputs

For each video, the following files are generated:
- `username_videoid.mp4` - Original video file
- `username_videoid.wav` - Extracted audio (if transcription enabled)
- `username_videoid.json` - Metadata including transcription data

## 🎯 Use Cases

### Video Processor
- **Multimodal AI Training**: Prepare video data for model training
- **Video Analysis**: Extract and analyze video content
- **Batch Processing**: Process large video datasets efficiently
- **Model Evaluation**: Test models on standardized video inputs

### TikTok Downloader
- 📚 **Content Analysis**: Analyze TikTok trends and topics
- 📝 **Research**: Academic studies on social media content
- 🎬 **Content Creation**: Extract quotes and highlights
- 📊 **Social Listening**: Monitor brand mentions and sentiment
- ♿ **Accessibility**: Generate captions for hearing impaired

## ⚡ Performance

### Video Processor
- **GPU Acceleration**: Automatic CUDA/MPS detection
- **Batch Processing**: Efficient multi-video processing
- **Memory Management**: Optimized memory usage
- **Backend Fallback**: Automatic fallback between video backends

### TikTok Transcription
- **GPU Processing**: 2-3x real-time on RTX 4090
- **Apple Silicon**: Optimized MPS backend support
- **CPU Fallback**: ~0.5x real-time processing
- **Model Caching**: First-run model downloads (~1-2GB)

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Running Tests
```bash
# Run video processor tests
python -m pytest video_processor/tests/

# Run TikTok downloader tests
cd tt_video_download && python -m pytest tests/
```

## 📋 Dependencies

### Core Dependencies
- **torch**: PyTorch for tensor operations
- **torchvision**: Video and image processing
- **PIL**: Image processing
- **numpy**: Numerical operations
- **requests**: HTTP requests

### Optional Dependencies
- **decord**: Alternative video backend
- **av**: PyAV video processing
- **ffmpeg-python**: Audio processing
- **transformers**: Whisper model support
- **pyannote.audio**: Speaker diarization

## 🐛 Troubleshooting

### Common Issues

1. **Video Loading Errors**:
   - Install multiple backends: `pip install decord av`
   - Check video format compatibility
   - Verify file paths and permissions

2. **GPU Memory Issues**:
   - Reduce batch size or frame count
   - Use CPU fallback if necessary
   - Monitor GPU memory usage

3. **TikTok Download Issues**:
   - Update yt-dlp: `pip install -U yt-dlp`
   - Check network connectivity
   - Verify URL format

4. **Transcription Issues**:
   - Activate conda environment: `conda activate python12`
   - Accept HuggingFace agreements for Pyannote
   - Check FFmpeg installation

### Performance Optimization
- Use GPU when available
- Enable model caching
- Batch process multiple videos
- Adjust configuration for your use case

## 📄 License

This project is for educational and research purposes. Please respect:
- Content creators' rights
- Platform terms of service
- Applicable laws and regulations

## 🤝 Support

- 📖 Check component-specific documentation in subdirectories
- 🐛 Report issues via GitHub Issues
- 💡 Suggest features via GitHub Discussions
- 📧 Contact maintainers for collaboration
