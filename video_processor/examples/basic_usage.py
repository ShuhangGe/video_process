"""
Basic Usage Example - Video Processing Pipeline

This example demonstrates the basic usage of the video processing pipeline
with different configurations and output formats.
"""

import sys
from pathlib import Path

# Add parent directory to path to import video_processor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from video_processor import VideoProcessor, get_default_config, get_vllm_config
from video_processor.config import get_fast_config, get_high_quality_config


def basic_video_processing():
    """Demonstrate basic video processing functionality."""
    print("=== Basic Video Processing Example ===")
    
    # Initialize with default configuration
    config = get_default_config()
    processor = VideoProcessor(config)
    
    # Example video file path (replace with actual video file)
    video_path = "sample_video.mp4"
    
    try:
        # Process video with default settings
        print(f"Processing video: {video_path}")
        result = processor.process(video_path)
        
        print(f"✅ Processing completed!")
        print(f"📊 Frames: {result.metadata['num_frames']}")
        print(f"📏 Resolution: {result.metadata['height']}x{result.metadata['width']}")
        print(f"⏱️  Total time: {result.timing['total_time']:.3f}s")
        
        if result.token_info:
            print(f"🔢 Total tokens: {result.token_info['total_tokens']}")
        
        return result
        
    except FileNotFoundError:
        print(f"❌ Video file not found: {video_path}")
        print("💡 Please provide a valid video file path")
        return None
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None


def configuration_examples():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Examples ===")
    
    # Fast processing configuration
    print("🚀 Fast Configuration:")
    fast_config = get_fast_config()
    fast_processor = VideoProcessor(fast_config)
    print(f"   - Max frames: {fast_config.sampling.max_frames}")
    print(f"   - Max pixels: {fast_config.processing.max_pixels}")
    
    # High quality configuration
    print("🎨 High Quality Configuration:")
    hq_config = get_high_quality_config()
    hq_processor = VideoProcessor(hq_config)
    print(f"   - Max frames: {hq_config.sampling.max_frames}")
    print(f"   - Max pixels: {hq_config.processing.max_pixels}")
    
    # VLLM configuration (if available)
    print("🤖 VLLM Configuration:")
    try:
        vllm_config = get_vllm_config()
        vllm_processor = VideoProcessor(vllm_config)
        print(f"   - VLLM enabled: {vllm_config.vllm.enable_vllm}")
        print(f"   - Model: {vllm_config.vllm.model_name}")
        print(f"   - Max model length: {vllm_config.vllm.max_model_len}")
    except Exception as e:
        print(f"   - VLLM not available: {e}")


def format_examples():
    """Demonstrate different output formats."""
    print("\n=== Output Format Examples ===")
    
    # Create a sample video tensor for demonstration
    sample_video = torch.randn(8, 3, 224, 224)  # 8 frames, 3 channels, 224x224
    
    config = get_default_config()
    processor = VideoProcessor(config)
    
    # Standard format
    print("📋 Standard Format:")
    standard_result = processor.format_handler.format_output(
        sample_video, 
        {"num_frames": 8, "height": 224, "width": 224},
        {"total_time": 1.0},
        format_type="standard"
    )
    print(f"   - Type: {type(standard_result).__name__}")
    print(f"   - Frames shape: {standard_result.frames.shape}")
    
    # HuggingFace format
    print("🤗 HuggingFace Format:")
    hf_result = processor.format_handler.format_output(
        sample_video,
        {"num_frames": 8, "height": 224, "width": 224},
        {"total_time": 1.0},
        format_type="huggingface"
    )
    print(f"   - Keys: {list(hf_result.keys())}")
    print(f"   - Pixel values shape: {hf_result['pixel_values'].shape}")
    
    # Raw format
    print("🔧 Raw Format:")
    raw_result = processor.format_handler.format_output(
        sample_video,
        {"num_frames": 8, "height": 224, "width": 224},
        {"total_time": 1.0},
        format_type="raw"
    )
    print(f"   - Keys: {list(raw_result.keys())}")


def token_calculation_example():
    """Demonstrate token calculation functionality."""
    print("\n=== Token Calculation Example ===")
    
    config = get_default_config()
    processor = VideoProcessor(config)
    
    # Create sample video tensors with different sizes
    videos = [
        torch.randn(4, 3, 224, 224),   # Small video
        torch.randn(16, 3, 448, 448),  # Medium video
        torch.randn(32, 3, 672, 672),  # Large video
    ]
    
    for i, video in enumerate(videos):
        token_info = processor.tokenizer.calculate_tokens(video, {})
        
        print(f"📹 Video {i+1}: {video.shape}")
        print(f"   - Total tokens: {token_info['total_tokens']}")
        print(f"   - Tokens per frame: {token_info['tokens_per_frame']}")
        print(f"   - Grid THW: {token_info['grid_thw']}")
        print(f"   - Memory estimate: {token_info['memory_info']['total_memory_mb']:.1f} MB")



def performance_info():
    """Display performance and system information."""
    print("\n=== Performance Information ===")
    
    config = get_default_config()
    processor = VideoProcessor(config)
    
    # Backend information
    backend_info = processor.frame_extractor.get_backend_info()
    print(f"🔧 Available backends: {backend_info['available_backends']}")
    print(f"🏃 Backend priority: {backend_info['backend_priority']}")
    
    # Processing information
    processing_info = processor.frame_processor.get_processing_info()
    print(f"🖼️  Image factor: {processing_info['image_factor']}")
    print(f"🔄 Interpolation: {processing_info['interpolation_mode']}")
    print(f"💾 Half precision: {processing_info['half_precision']}")
    
    # Tokenizer information
    tokenizer_info = processor.tokenizer.get_tokenizer_info()
    print(f"🔤 Patch size: {tokenizer_info['patch_size']}")
    print(f"🎯 Tokens per patch: {tokenizer_info['tokens_per_patch']}")


def main():
    """Run all examples."""
    print("🎬 Video Processor - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    basic_video_processing()
    configuration_examples()
    format_examples()
    token_calculation_example()
    performance_info()
    
    print("\n✨ All examples completed!")
    print("\n💡 Tips:")
    print("   - Replace 'sample_video.mp4' with your actual video file")
    print("   - Check the configuration options for optimization")
    print("   - Enable VLLM integration for serving capabilities")


if __name__ == "__main__":
    main() 