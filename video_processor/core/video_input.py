"""
Video Input Handler - Handles various video input formats and normalizes them.

This module provides comprehensive video input handling, supporting local files,
URLs, base64-encoded videos, byte streams, and pre-extracted frame sequences.
"""

import os
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlparse
import base64

import requests
from PIL import Image


logger = logging.getLogger(__name__)


class VideoInputHandler:
    """
    Handles various video input formats and normalizes them for processing.
    
    Supports:
    - Local video files (MP4, MOV, AVI, etc.)
    - URLs (HTTP/HTTPS)
    - Base64 encoded videos
    - Pre-extracted frame sequences
    - Video byte streams
    - Configuration dictionaries
    """
    
    def __init__(self, config):
        """Initialize the video input handler with configuration."""
        self.config = config
        self.temp_dir = None
        self._setup_temp_dir()
    
    def _setup_temp_dir(self):
        """Setup temporary directory for downloaded/converted files."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_processor_"))
        logger.debug(f"Created temporary directory: {self.temp_dir}")
    
    def process_input(self, video_input: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Process video input and return normalized configuration.
        
        Args:
            video_input: Various input formats:
                - str: File path, URL, or base64 data
                - dict: Configuration dictionary
                - list: Pre-extracted frames
        
        Returns:
            Normalized video configuration dictionary
        """
        start_time = time.time()
        
        try:
            if isinstance(video_input, str):
                config = self._process_string_input(video_input)
            elif isinstance(video_input, dict):
                config = self._process_dict_input(video_input)
            elif isinstance(video_input, list):
                config = self._process_frame_list_input(video_input)
            else:
                raise ValueError(f"Unsupported video input type: {type(video_input)}")
            
            # Add metadata
            config["input_processing_time"] = time.time() - start_time
            config["input_hash"] = self._generate_input_hash(video_input)
            
            # Validate the configuration
            self._validate_config(config)
            
            logger.info(f"Video input processed in {config['input_processing_time']:.3f}s")
            return config
            
        except Exception as e:
            logger.error(f"Failed to process video input: {e}")
            if self.config.strict_mode:
                raise
            else:
                # Return a fallback configuration
                return self._create_fallback_config(video_input, str(e))
    
    def _process_string_input(self, video_input: str) -> Dict[str, Any]:
        """Process string input (file path, URL, or base64)."""
        if video_input.startswith("http://") or video_input.startswith("https://"):
            return self._process_url_input(video_input)
        elif video_input.startswith("data:video"):
            return self._process_base64_input(video_input)
        elif video_input.startswith("file://"):
            return self._process_file_input(video_input[7:])
        else:
            # Assume it's a local file path
            return self._process_file_input(video_input)
    
    def _process_file_input(self, file_path: str) -> Dict[str, Any]:
        """Process local file input."""
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Get file metadata
        file_size = file_path.stat().st_size
        file_ext = file_path.suffix.lower()
        
        # Validate file extension
        supported_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp', '.flv'}
        if file_ext not in supported_extensions:
            logger.warning(f"Potentially unsupported video format: {file_ext}")
        
        logger.debug(f"Processing local video file: {file_path} ({file_size} bytes)")
        
        return {
            "video": str(file_path),
            "input_type": "local_file",
            "file_size": file_size,
            "file_extension": file_ext,
            "original_path": str(file_path)
        }
    
    def _process_url_input(self, url: str) -> Dict[str, Any]:
        """Process URL input by downloading the video."""
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        # Generate filename from URL
        filename = self._generate_filename_from_url(url)
        local_path = self.temp_dir / filename
        
        # Download the video
        logger.info(f"Downloading video from URL: {url}")
        download_start = time.time()
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('video/'):
                logger.warning(f"Content type may not be video: {content_type}")
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            download_time = time.time() - download_start
            logger.info(f"Downloaded {downloaded_size} bytes in {download_time:.3f}s")
            
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download video from {url}: {e}")
        
        return {
            "video": str(local_path),
            "input_type": "url",
            "original_url": url,
            "download_time": download_time,
            "file_size": downloaded_size,
            "content_type": content_type
        }
    
    def _process_base64_input(self, base64_data: str) -> Dict[str, Any]:
        """Process base64 encoded video input."""
        try:
            # Parse base64 data URL
            if "base64," in base64_data:
                header, data = base64_data.split("base64,", 1)
                # Extract mime type from header
                mime_type = header.replace("data:", "").replace(";", "")
            else:
                data = base64_data
                mime_type = "video/mp4"  # Default assumption
            
            # Decode base64 data
            video_bytes = base64.b64decode(data)
            
            # Generate filename
            extension = self._get_extension_from_mime_type(mime_type)
            filename = f"video_{int(time.time())}{extension}"
            local_path = self.temp_dir / filename
            
            # Save to temporary file
            with open(local_path, 'wb') as f:
                f.write(video_bytes)
            
            logger.debug(f"Decoded base64 video: {len(video_bytes)} bytes")
            
            return {
                "video": str(local_path),
                "input_type": "base64",
                "file_size": len(video_bytes),
                "mime_type": mime_type,
                "decoded_size": len(video_bytes)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 video data: {e}")
    
    def _process_frame_list_input(self, frame_list: List) -> Dict[str, Any]:
        """Process pre-extracted frame list input."""
        if not frame_list:
            raise ValueError("Frame list cannot be empty")
        
        processed_frames = []
        total_size = 0
        
        for i, frame in enumerate(frame_list):
            try:
                if isinstance(frame, str):
                    # Frame is a file path or URL
                    frame_path = self._process_frame_path(frame, i)
                    processed_frames.append(frame_path)
                elif isinstance(frame, Image.Image):
                    # Frame is a PIL Image
                    frame_path = self._save_pil_image(frame, i)
                    processed_frames.append(frame_path)
                    total_size += self._estimate_image_size(frame)
                else:
                    raise ValueError(f"Unsupported frame type: {type(frame)}")
                    
            except Exception as e:
                logger.warning(f"Failed to process frame {i}: {e}")
                if self.config.strict_mode:
                    raise
        
        if not processed_frames:
            raise ValueError("No valid frames found in frame list")
        
        logger.info(f"Processed {len(processed_frames)} frames")
        
        return {
            "video": processed_frames,
            "input_type": "frame_list",
            "num_frames": len(processed_frames),
            "total_estimated_size": total_size
        }
    
    def _process_dict_input(self, config_dict: Dict) -> Dict[str, Any]:
        """Process configuration dictionary input."""
        config = config_dict.copy()
        
        # Ensure required fields
        if "video" not in config:
            raise ValueError("Configuration dictionary must contain 'video' key")
        
        # Process the video field recursively if it's not a string
        if not isinstance(config["video"], str):
            video_config = self.process_input(config["video"])
            config.update(video_config)
        else:
            # Process string video field
            video_config = self._process_string_input(config["video"])
            config.update(video_config)
        
        # Set input type
        config["input_type"] = "configuration"
        
        # Validate time range if specified
        if "video_start" in config and "video_end" in config:
            if config["video_start"] >= config["video_end"]:
                raise ValueError("video_start must be less than video_end")
        
        return config
    
    def _process_frame_path(self, frame_path: str, index: int) -> str:
        """Process individual frame path."""
        if frame_path.startswith("http://") or frame_path.startswith("https://"):
            # Download frame from URL
            filename = f"frame_{index}_{int(time.time())}.jpg"
            local_path = self.temp_dir / filename
            
            response = requests.get(frame_path, timeout=10)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return str(local_path)
        
        elif frame_path.startswith("file://"):
            return frame_path[7:]
        
        else:
            # Assume local file path
            frame_path = Path(frame_path)
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            return str(frame_path)
    
    def _save_pil_image(self, image: Image.Image, index: int) -> str:
        """Save PIL Image to temporary file."""
        filename = f"frame_{index}_{int(time.time())}.jpg"
        local_path = self.temp_dir / filename
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        image.save(local_path, 'JPEG', quality=95)
        return str(local_path)
    
    def _generate_filename_from_url(self, url: str) -> str:
        """Generate filename from URL."""
        parsed = urlparse(url)
        path = Path(parsed.path)
        
        if path.suffix:
            # Use original filename if available
            filename = path.name
        else:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"video_{url_hash}.mp4"
        
        return filename
    
    def _get_extension_from_mime_type(self, mime_type: str) -> str:
        """Get file extension from MIME type."""
        mime_to_ext = {
            "video/mp4": ".mp4",
            "video/mov": ".mov",
            "video/avi": ".avi",
            "video/mkv": ".mkv",
            "video/webm": ".webm",
            "video/quicktime": ".mov",
            "video/x-msvideo": ".avi"
        }
        return mime_to_ext.get(mime_type, ".mp4")
    
    def _estimate_image_size(self, image: Image.Image) -> int:
        """Estimate image size in bytes."""
        width, height = image.size
        channels = len(image.getbands())
        return width * height * channels  # Rough estimate
    
    def _generate_input_hash(self, video_input: Any) -> str:
        """Generate hash for input for caching purposes."""
        input_str = str(video_input)
        if len(input_str) > 1000:
            input_str = input_str[:500] + "..." + input_str[-500:]
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the processed configuration."""
        required_keys = ["video", "input_type"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in configuration: {key}")
        
        # Validate video path/list
        if config["input_type"] == "frame_list":
            if not isinstance(config["video"], list) or not config["video"]:
                raise ValueError("Frame list cannot be empty")
        else:
            video_path = config["video"]
            if isinstance(video_path, str) and not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def _create_fallback_config(self, original_input: Any, error_msg: str) -> Dict[str, Any]:
        """Create a fallback configuration when processing fails."""
        logger.warning(f"Creating fallback configuration due to error: {error_msg}")
        
        return {
            "video": None,
            "input_type": "fallback",
            "error": error_msg,
            "original_input": str(original_input)[:500],  # Truncate for safety
            "fallback": True
        }
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()


class VideoInputValidator:
    """Validates video inputs and provides detailed feedback."""
    
    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """Validate file path and return (is_valid, message)."""
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"File does not exist: {file_path}"
            if not path.is_file():
                return False, f"Path is not a file: {file_path}"
            if path.stat().st_size == 0:
                return False, f"File is empty: {file_path}"
            return True, "Valid file path"
        except Exception as e:
            return False, f"Invalid file path: {e}"
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Validate URL and return (is_valid, message)."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP/HTTPS URLs are supported"
            return True, "Valid URL"
        except Exception as e:
            return False, f"Invalid URL: {e}"
    
    @staticmethod
    def validate_base64(base64_data: str) -> Tuple[bool, str]:
        """Validate base64 data and return (is_valid, message)."""
        try:
            if "base64," in base64_data:
                _, data = base64_data.split("base64,", 1)
            else:
                data = base64_data
            
            decoded = base64.b64decode(data)
            if len(decoded) == 0:
                return False, "Base64 data decodes to empty content"
            
            return True, f"Valid base64 data ({len(decoded)} bytes)"
        except Exception as e:
            return False, f"Invalid base64 data: {e}"
    
    @classmethod
    def validate_input(cls, video_input: Any) -> Tuple[bool, str]:
        """Validate any video input and return (is_valid, message)."""
        if isinstance(video_input, str):
            if video_input.startswith(("http://", "https://")):
                return cls.validate_url(video_input)
            elif video_input.startswith("data:video"):
                return cls.validate_base64(video_input)
            else:
                return cls.validate_file_path(video_input)
        
        elif isinstance(video_input, dict):
            if "video" not in video_input:
                return False, "Configuration dictionary must contain 'video' key"
            return cls.validate_input(video_input["video"])
        
        elif isinstance(video_input, list):
            if not video_input:
                return False, "Frame list cannot be empty"
            # Validate first few frames as sample
            for i, frame in enumerate(video_input[:3]):
                if isinstance(frame, str):
                    valid, msg = cls.validate_input(frame)
                    if not valid:
                        return False, f"Invalid frame {i}: {msg}"
                elif not isinstance(frame, Image.Image):
                    return False, f"Unsupported frame type at index {i}: {type(frame)}"
            return True, f"Valid frame list ({len(video_input)} frames)"
        
        else:
            return False, f"Unsupported input type: {type(video_input)}" 