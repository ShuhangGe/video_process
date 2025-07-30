# TikTok Video Downloader with Audio Transcription

An enhanced TikTok video downloader that not only downloads videos but also extracts audio and provides automatic speech-to-text transcription with speaker identification using OpenAI's Whisper and Pyannote.

## Features

### Core Features
- ✅ **Video Downloading**: Download TikTok videos from URLs
- ✅ **Metadata Extraction**: Save detailed video information (title, uploader, stats, etc.)
- ✅ **Batch Processing**: Process multiple URLs from a file
- ✅ **Progress Tracking**: Real-time download progress and statistics

### New Audio Transcription Features
- 🎵 **Audio Extraction**: Extract audio from videos using FFmpeg
- 🎤 **Speech Recognition**: Convert speech to text using Whisper Large-v3 Turbo
- 👥 **Speaker Diarization**: Identify different speakers using Pyannote
- ⏰ **Timestamp Alignment**: Precise timing for each speech segment
- 🔗 **Smart Merging**: Combine consecutive segments from same speaker
- 📊 **Enhanced Metadata**: Rich JSON output with transcription data

## Installation

### Quick Setup (Recommended)
```bash
cd video_process
python setup_transcription.py
```

### Manual Setup
1. **Install conda environment:**
```bash
conda create -n python12 python=3.11 -y
conda activate python12
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

3. **Setup HuggingFace (for speaker diarization):**
   - Create account at [HuggingFace](https://huggingface.co)
   - Accept user agreement at [Pyannote Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization)
   - Get access token from [Settings](https://huggingface.co/settings/tokens)

## Usage

### Basic Usage
1. **Create URL file:**
```bash
# Create urls.txt with TikTok URLs (one per line)
echo "https://www.tiktok.com/@username/video/1234567890" > urls.txt
```

2. **Run the downloader:**
```bash
conda activate python12  # or python11
python get_tt.py
```

### Configuration
Edit the main section in `get_tt.py`:
```python
# Configuration
urls_file = "urls.txt"
output_directory = "/home/shuhang/code/video_process/tt_videos"
enable_transcription = True  # Set to False to skip audio processing
```

### File Structure
After processing, you'll get:
```
tt_videos/
├── username1/
│   ├── username1_videoid1.mp4      # Video file
│   ├── username1_videoid1.wav      # Extracted audio
│   └── username1_videoid1.json     # Metadata + transcription
└── username2/
    ├── username2_videoid2.mp4
    ├── username2_videoid2.wav
    └── username2_videoid2.json
```

## Output Format

### Enhanced JSON Metadata
```json
{
  "title": "Video Title",
  "url": "https://tiktok.com/@user/video/123",
  "uploader": "username",
  "duration_seconds": 30,
  "view_count": 1000000,
  "transcription": {
    "full_text": "Complete transcription text...",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello everyone!",
        "speaker": "SPEAKER_00"
      },
      {
        "start": 2.5,
        "end": 5.0,
        "text": "Welcome to my video",
        "speaker": "SPEAKER_00"
      }
    ],
    "total_segments": 2,
    "processing_date": "2025-01-15T10:30:00"
  },
  "audio_info": {
    "file_path": "/path/to/audio.wav",
    "file_size_mb": 2.1
  },
  "processing_settings": {
    "whisper_model": "openai/whisper-large-v3-turbo",
    "pyannote_model": "pyannote/speaker-diarization-3.1",
    "device_used": "cuda:0",
    "diarization_enabled": true
  }
}
```

## Technical Details

### Audio Processing Pipeline
1. **Video Download**: Download MP4 using yt-dlp
2. **Audio Extraction**: Convert to WAV (16kHz, mono) using FFmpeg
3. **Speech Recognition**: Process with Whisper for transcription + timestamps
4. **Speaker Diarization**: Identify speakers using Pyannote neural networks
5. **Alignment**: Match transcription segments with speaker labels
6. **Merging**: Combine consecutive segments from same speaker

### Models Used
- **Whisper Large-v3 Turbo**: Fast, accurate multilingual speech recognition
- **Pyannote Speaker Diarization 3.1**: State-of-the-art speaker identification
- **Hardware Acceleration**: Automatic GPU/MPS/CPU detection

### Performance
- **GPU**: Fastest processing (RTX 4090: ~2-3x real-time)
- **Apple Silicon**: Good performance with MPS backend
- **CPU**: Slower but functional (~0.5x real-time)

## Troubleshooting

### Common Issues

1. **Import Errors**:
```bash
# Activate conda environment first
conda activate python12
```

2. **Pyannote Access Denied**:
   - Accept user agreement at HuggingFace
   - Check your access token

3. **CUDA Out of Memory**:
   - The script automatically falls back to CPU
   - Consider using smaller batch sizes

4. **FFmpeg Not Found**:
```bash
conda install -c conda-forge ffmpeg -y
```

### Performance Tips
- Use GPU when available for faster processing
- First run downloads models (~1-2GB) - be patient
- Enable transcription only when needed (`enable_transcription = False`)
- Process smaller batches if memory is limited

## Environment Variables
```bash
# Optional: Set HuggingFace token
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## Dependencies
- **Core**: yt-dlp, ffmpeg-python
- **AI/ML**: torch, transformers, pyannote.audio
- **Audio**: librosa, soundfile, numpy

## Features Comparison

| Feature | Basic Version | Enhanced Version |
|---------|---------------|------------------|
| Video Download | ✅ | ✅ |
| Metadata Extraction | ✅ | ✅ |
| Audio Extraction | ❌ | ✅ |
| Speech-to-Text | ❌ | ✅ |
| Speaker ID | ❌ | ✅ |
| Timestamp Alignment | ❌ | ✅ |
| Multilingual Support | ❌ | ✅ |

## Use Cases
- 📚 **Content Analysis**: Analyze TikTok trends and topics
- 📝 **Research**: Academic studies on social media content
- 🎬 **Content Creation**: Extract quotes and highlights
- 📊 **Social Listening**: Monitor brand mentions and sentiment
- ♿ **Accessibility**: Generate captions for hearing impaired

## License
This project is for educational and research purposes. Respect TikTok's terms of service and content creators' rights.