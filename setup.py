#!/usr/bin/env python3
"""
Setup script for Video Processing Pipeline
Helps with installation and environment configuration
"""

import subprocess
import sys
import os
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_system():
    """Detect the operating system and architecture"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    logger.info(f"🖥️  Detected system: {system} ({machine})")
    
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def run_command(command, description, check=True):
    """Run a shell command and handle errors"""
    try:
        logger.info(f"🔄 {description}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            logger.warning(f"⚠️  {description} completed with warnings")
            logger.warning(f"Output: {result.stderr}")
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {str(e)}")
        return False, str(e)

def check_gpu_support():
    """Check for GPU support"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 CUDA GPU detected: {gpu_name}")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("🍎 Apple Silicon MPS detected")
            return "mps"
        else:
            logger.info("💻 Using CPU (GPU acceleration not available)")
            return "cpu"
    except ImportError:
        logger.info("⏳ PyTorch not installed yet - will check GPU after installation")
        return "unknown"

def install_core_dependencies():
    """Install core project dependencies"""
    logger.info("📦 Installing core dependencies...")
    
    # Core dependencies that should work on all systems
    core_deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "packaging>=21.0",
        "requests>=2.28.0"
    ]
    
    for dep in core_deps:
        success, output = run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}", check=False)
        if not success:
            logger.warning(f"⚠️  Failed to install {dep}, continuing...")
    
    return True

def install_optional_dependencies():
    """Install optional dependencies"""
    logger.info("📦 Installing optional dependencies...")
    
    # Ask user what they want to install
    print("\n" + "="*60)
    print("OPTIONAL COMPONENTS")
    print("="*60)
    
    # TikTok downloader with transcription
    response = input("Install TikTok downloader with audio transcription? (y/N): ").lower()
    if response.startswith('y'):
        tiktok_deps = [
            "yt-dlp>=2023.12.30",
            "ffmpeg-python>=0.2.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0"
        ]
        
        for dep in tiktok_deps:
            run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}", check=False)
        
        # Pyannote requires special handling
        response = input("Install Pyannote for speaker diarization? Requires HuggingFace account (y/N): ").lower()
        if response.startswith('y'):
            run_command("pip install 'pyannote.audio>=3.1.0'", "Installing Pyannote", check=False)
    
    # Additional video backends
    response = input("Install additional video backends (decord, av)? (y/N): ").lower()
    if response.startswith('y'):
        run_command("pip install decord av", "Installing additional video backends", check=False)
    
    return True

def install_ffmpeg():
    """Install FFmpeg system dependency"""
    system = detect_system()
    
    logger.info("🎬 Checking FFmpeg installation...")
    
    # Check if FFmpeg is already installed
    success, _ = run_command("ffmpeg -version", "Checking FFmpeg", check=False)
    if success:
        logger.info("✅ FFmpeg is already installed")
        return True
    
    logger.info("FFmpeg not found. Installation instructions:")
    print("\n" + "="*60)
    print("FFMPEG INSTALLATION")
    print("="*60)
    
    if system == "linux":
        print("Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("CentOS/RHEL:   sudo yum install ffmpeg")
        print("Fedora:        sudo dnf install ffmpeg")
    elif system == "macos":
        print("With Homebrew: brew install ffmpeg")
        print("With MacPorts: sudo port install ffmpeg")
    elif system == "windows":
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Add to system PATH")
        print("3. Or use: winget install ffmpeg")
    
    print("Alternative: conda install -c conda-forge ffmpeg")
    print("="*60)
    
    response = input("\nHave you installed FFmpeg? (y/N): ").lower()
    if response.startswith('y'):
        success, _ = run_command("ffmpeg -version", "Verifying FFmpeg installation", check=False)
        return success
    
    return False

def create_example_config():
    """Create example configuration files"""
    logger.info("📋 Creating example configuration files...")
    
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Basic video processing example
    basic_example = '''#!/usr/bin/env python3
"""
Basic Video Processing Example
"""

from video_processor import VideoProcessor, get_default_config

def main():
    # Initialize processor with default config
    config = get_default_config()
    processor = VideoProcessor(config)
    
    # Process a single video
    video_path = "path/to/your/video.mp4"
    result = processor.process_video(video_path)
    
    print(f"Processed {len(result['frames'])} frames")
    print(f"Output shape: {result['frames'].shape}")
    print(f"Token count: {result['token_count']}")

if __name__ == "__main__":
    main()
'''
    
    with open(examples_dir / "basic_video_processing.py", "w") as f:
        f.write(basic_example)
    
    logger.info("✅ Example files created in examples/ directory")
    return True

def run_test():
    """Run basic functionality test"""
    logger.info("🧪 Running basic functionality test...")
    
    test_script = '''
import sys
try:
    # Test core imports
    import torch
    import torchvision
    import numpy as np
    from PIL import Image
    print("✅ Core dependencies imported successfully")
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available")
    else:
        print("ℹ️  Using CPU (slower but functional)")
    
    # Test video processor import
    try:
        from video_processor import VideoProcessor
        print("✅ Video processor available")
    except ImportError:
        print("ℹ️  Video processor module not in path (expected for basic setup)")
    
    print("✅ Basic functionality test passed")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
    
    # Write test script to temporary file
    with open("/tmp/test_video_processor.py", "w") as f:
        f.write(test_script)
    
    # Run test
    success, output = run_command("python /tmp/test_video_processor.py", "Running functionality test", check=False)
    
    if success:
        logger.info("🎉 All tests passed!")
        print(output)
        return True
    else:
        logger.warning("⚠️  Some tests failed, but basic functionality should work")
        print(output)
        return False

def main():
    """Main setup function"""
    print("🚀 Video Processing Pipeline Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect system
    system = detect_system()
    
    # Install core dependencies
    if not install_core_dependencies():
        logger.error("❌ Failed to install core dependencies")
        sys.exit(1)
    
    # Check GPU support
    gpu_type = check_gpu_support()
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Install FFmpeg
    install_ffmpeg()
    
    # Create example files
    create_example_config()
    
    # Run tests
    run_test()
    
    # Final instructions
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Check examples/ directory for usage examples")
    print("2. For TikTok downloader: cd tt_video_download && python setup_transcription.py")
    print("3. Read README.md for detailed documentation")
    print("4. Start with: python examples/basic_video_processing.py")
    print("="*60)

if __name__ == "__main__":
    main()