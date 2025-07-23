"""
Format Handler - Multiple output format support for video processing results.

This module provides comprehensive format handling capabilities to convert video
processing results into different formats compatible with various systems and frameworks.
"""

import base64
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, asdict
from io import BytesIO

import torch
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class ProcessedVideo:
    """Standard output format for processed video data."""
    
    # Video data
    frames: Union[torch.Tensor, List[Image.Image], np.ndarray]
    metadata: Dict[str, Any]
    timing: Dict[str, float]
    
    # Optional fields
    token_info: Optional[Dict[str, Any]] = None
    quality_info: Optional[Dict[str, Any]] = None
    cache_info: Optional[Dict[str, Any]] = None


class FormatHandler:
    """
    Handles conversion between different output formats.
    
    Supported formats:
    - Standard: ProcessedVideo dataclass
    - HuggingFace: Compatible with transformers pipeline
    - OpenAI: Base64 encoded for API consumption  
    - Streaming: Iterator for large video processing
    - Raw: Direct tensor/array access
    """
    
    def __init__(self, config):
        """Initialize format handler with configuration."""
        self.config = config
        self.output_config = config.output
        
        logger.debug(f"FormatHandler initialized with default format: {self.output_config.default_format}")
    
    def format_output(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                     timing: Dict[str, float], format_type: str = None,
                     **additional_data) -> Any:
        """
        Format video processing output according to specified format.
        
        Args:
            video_tensor: Processed video tensor (T, C, H, W)
            metadata: Video metadata
            timing: Processing timing information
            format_type: Output format type
            **additional_data: Additional data (token_info, quality_info, etc.)
            
        Returns:
            Formatted output according to format_type
        """
        format_type = format_type or self.output_config.default_format
        
        # Add timing and metadata if configured
        if self.output_config.include_timing:
            timing.update({"format_time": time.time()})
        
        if self.output_config.include_metadata:
            metadata.update({
                "format_type": format_type,
                "tensor_shape": list(video_tensor.shape) if video_tensor is not None else None,
                "tensor_dtype": str(video_tensor.dtype) if video_tensor is not None else None,
            })
        
        # Route to specific formatter
        if format_type == "standard":
            return self._format_standard(video_tensor, metadata, timing, **additional_data)
        elif format_type == "huggingface":
            return self._format_huggingface(video_tensor, metadata, timing, **additional_data)
        elif format_type == "openai":
            return self._format_openai(video_tensor, metadata, timing, **additional_data)
        elif format_type == "streaming":
            return self._format_streaming(video_tensor, metadata, timing, **additional_data)
        elif format_type == "raw":
            return self._format_raw(video_tensor, metadata, timing, **additional_data)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_standard(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                        timing: Dict[str, float], **additional_data) -> ProcessedVideo:
        """Format as standard ProcessedVideo object."""
        # Convert tensor based on configuration
        if self.output_config.standard_format.get("tensor_dtype") == "float16":
            video_tensor = video_tensor.half()
        elif self.output_config.standard_format.get("tensor_dtype") == "float32":
            video_tensor = video_tensor.float()
        
        # Add standard metadata
        if self.output_config.standard_format.get("include_fps", True):
            metadata.setdefault("sample_fps", metadata.get("video_fps", 30.0))
        
        if self.output_config.standard_format.get("include_grid_thw", True):
            if video_tensor is not None:
                t, c, h, w = video_tensor.shape
                # Calculate grid dimensions for positional encoding
                image_factor = self.config.processing.image_factor
                grid_thw = [t, h // image_factor, w // image_factor]
                metadata["grid_thw"] = grid_thw
        
        return ProcessedVideo(
            frames=video_tensor,
            metadata=metadata,
            timing=timing,
            token_info=additional_data.get("token_info"),
            quality_info=additional_data.get("quality_info"),
            cache_info=additional_data.get("cache_info")
        )
    
    def _format_huggingface(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                           timing: Dict[str, float], **additional_data) -> Dict[str, Any]:
        """Format for HuggingFace transformers compatibility."""
        hf_config = self.output_config.huggingface_format
        
        # Convert to format expected by HuggingFace processors
        output = {
            "pixel_values": video_tensor,
            "pixel_values_videos": video_tensor,
        }
        
        # Add grid information for positional encoding
        if video_tensor is not None:
            t, c, h, w = video_tensor.shape
            image_factor = self.config.processing.image_factor
            
            # Image grid (for each frame)
            image_grid_thw = torch.tensor([[1, h // image_factor, w // image_factor]] * t)
            output["image_grid_thw"] = image_grid_thw
            
            # Video grid (overall)
            video_grid_thw = torch.tensor([[t, h // image_factor, w // image_factor]])
            output["video_grid_thw"] = video_grid_thw
        
        # Add processor class information
        if hf_config.get("processor_class"):
            output["processor_class"] = hf_config["processor_class"]
        
        # Add metadata if configured
        if self.output_config.include_metadata:
            output["metadata"] = metadata
        
        if self.output_config.include_timing:
            output["timing"] = timing
        
        # Return tensors format
        return_tensors = hf_config.get("return_tensors", "pt")
        if return_tensors == "pt":
            # Already torch tensors
            pass
        elif return_tensors == "np":
            # Convert to numpy
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output[key] = value.numpy()
        
        logger.debug(f"Formatted as HuggingFace format with {len(output)} keys")
        return output
    
    def _format_openai(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                      timing: Dict[str, float], **additional_data) -> Dict[str, Any]:
        """Format for OpenAI API compatibility."""
        openai_config = self.output_config.openai_format
        
        # Convert frames to base64 if configured
        if openai_config.get("base64_encode", True):
            frames_data = self._tensor_to_base64_frames(video_tensor)
        else:
            # Convert to list of arrays
            frames_data = [frame.numpy() for frame in video_tensor]
        
        output = {
            "frames": frames_data,
            "num_frames": len(frames_data) if frames_data else 0,
            "format": "base64" if openai_config.get("base64_encode", True) else "array"
        }
        
        # Add usage information if configured
        if openai_config.get("include_usage", True):
            token_info = additional_data.get("token_info", {})
            output["usage"] = {
                "total_tokens": token_info.get("total_tokens", 0),
                "processing_time_ms": timing.get("total_time", 0) * 1000,
                "frames_processed": metadata.get("num_frames", 0)
            }
        
        # Add metadata
        if self.output_config.include_metadata:
            output["metadata"] = metadata
        
        logger.debug(f"Formatted as OpenAI format with {output['num_frames']} frames")
        return output
    
    def _format_streaming(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                         timing: Dict[str, float], **additional_data) -> Iterator[Dict[str, Any]]:
        """Format as streaming iterator for large videos."""
        streaming_config = self.output_config.streaming_format
        chunk_size = streaming_config.get("chunk_size", 4)
        overlap = streaming_config.get("overlap", 1)
        
        if video_tensor is None or video_tensor.numel() == 0:
            return iter([])
        
        def stream_generator():
            total_frames = video_tensor.shape[0]
            start_idx = 0
            chunk_idx = 0
            
            while start_idx < total_frames:
                end_idx = min(start_idx + chunk_size, total_frames)
                chunk_tensor = video_tensor[start_idx:end_idx]
                
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_idx": chunk_idx,
                    "chunk_start": start_idx,
                    "chunk_end": end_idx,
                    "chunk_size": end_idx - start_idx,
                    "total_chunks": (total_frames + chunk_size - 1) // (chunk_size - overlap),
                    "is_final_chunk": end_idx >= total_frames
                })
                
                # Create chunk timing
                chunk_timing = timing.copy()
                chunk_timing["chunk_time"] = time.time()
                
                yield {
                    "frames": chunk_tensor,
                    "metadata": chunk_metadata,
                    "timing": chunk_timing,
                    "token_info": additional_data.get("token_info"),
                }
                
                # Move to next chunk with overlap
                start_idx = end_idx - overlap
                if start_idx >= end_idx:  # Avoid infinite loop
                    break
                chunk_idx += 1
        
        logger.debug(f"Created streaming iterator for {video_tensor.shape[0]} frames")
        return stream_generator()
    
    def _format_raw(self, video_tensor: torch.Tensor, metadata: Dict[str, Any],
                   timing: Dict[str, float], **additional_data) -> Dict[str, Any]:
        """Format as raw data with minimal processing."""
        return {
            "video_tensor": video_tensor,
            "metadata": metadata,
            "timing": timing,
            **additional_data
        }
    
    def _tensor_to_base64_frames(self, video_tensor: torch.Tensor) -> List[str]:
        """Convert video tensor to list of base64 encoded frame strings."""
        if video_tensor is None or video_tensor.numel() == 0:
            return []
        
        base64_frames = []
        
        for frame_tensor in video_tensor:
            # Convert tensor to PIL Image
            # Assume tensor is in [0, 1] range
            frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame_image = Image.fromarray(frame_np)
            
            # Convert to base64
            buffer = BytesIO()
            frame_image.save(buffer, format='JPEG', quality=95)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_frames.append(f"data:image/jpeg;base64,{frame_base64}")
        
        return base64_frames
    
    def convert_format(self, processed_video: ProcessedVideo, target_format: str) -> Any:
        """
        Convert ProcessedVideo to different format.
        
        Args:
            processed_video: ProcessedVideo object to convert
            target_format: Target format type
            
        Returns:
            Converted output in target format
        """
        return self.format_output(
            video_tensor=processed_video.frames,
            metadata=processed_video.metadata,
            timing=processed_video.timing,
            format_type=target_format,
            token_info=processed_video.token_info,
            quality_info=processed_video.quality_info,
            cache_info=processed_video.cache_info
        )
    
    def batch_format(self, video_tensors: List[torch.Tensor], 
                    metadata_list: List[Dict[str, Any]], 
                    timing_list: List[Dict[str, float]],
                    format_type: str = None,
                    **additional_data_list) -> List[Any]:
        """
        Format multiple videos in batch.
        
        Args:
            video_tensors: List of video tensors
            metadata_list: List of metadata dictionaries
            timing_list: List of timing dictionaries
            format_type: Output format type
            **additional_data_list: Additional data lists
            
        Returns:
            List of formatted outputs
        """
        if not (len(video_tensors) == len(metadata_list) == len(timing_list)):
            raise ValueError("All input lists must have the same length")
        
        formatted_outputs = []
        
        for i, (video_tensor, metadata, timing) in enumerate(zip(video_tensors, metadata_list, timing_list)):
            # Extract additional data for this video
            additional_data = {}
            for key, value_list in additional_data_list.items():
                if isinstance(value_list, list) and i < len(value_list):
                    additional_data[key] = value_list[i]
            
            formatted_output = self.format_output(
                video_tensor=video_tensor,
                metadata=metadata,
                timing=timing,
                format_type=format_type,
                **additional_data
            )
            formatted_outputs.append(formatted_output)
        
        logger.info(f"Batch formatted {len(formatted_outputs)} videos to {format_type or self.output_config.default_format}")
        return formatted_outputs
    
    def get_format_info(self) -> Dict[str, Any]:
        """Get information about supported formats and current configuration."""
        return {
            "supported_formats": ["standard", "huggingface", "openai", "streaming", "raw"],
            "default_format": self.output_config.default_format,
            "include_timing": self.output_config.include_timing,
            "include_metadata": self.output_config.include_metadata,
            "include_debug_info": self.output_config.include_debug_info,
            "format_configs": {
                "standard": self.output_config.standard_format,
                "huggingface": self.output_config.huggingface_format,
                "openai": self.output_config.openai_format,
                "streaming": self.output_config.streaming_format,
            }
        }
    
    def validate_format_compatibility(self, format_type: str, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Validate if video tensor is compatible with target format.
        
        Args:
            format_type: Target format type
            video_tensor: Video tensor to validate
            
        Returns:
            Validation results
        """
        validation = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        if video_tensor is None or video_tensor.numel() == 0:
            validation["compatible"] = False
            validation["warnings"].append("Empty video tensor")
            return validation
        
        t, c, h, w = video_tensor.shape
        
        # Format-specific validations
        if format_type == "openai":
            # Check if suitable for base64 encoding
            if t > 100:
                validation["warnings"].append(f"Large number of frames ({t}) may result in large base64 output")
                validation["recommendations"].append("Consider using streaming format for large videos")
            
            total_pixels = t * h * w * c
            if total_pixels > 100 * 1024 * 1024:  # 100M pixels
                validation["warnings"].append("Very large video may cause memory issues with base64 encoding")
        
        elif format_type == "huggingface":
            # Check tensor format
            if c != 3:
                validation["warnings"].append(f"Unexpected number of channels ({c}), HuggingFace typically expects 3")
            
            # Check dimensions
            if h % 28 != 0 or w % 28 != 0:
                validation["warnings"].append("Dimensions not aligned to image_factor (28), may cause issues with tokenization")
        
        elif format_type == "streaming":
            # Check if beneficial for streaming
            if t < 8:
                validation["recommendations"].append("Few frames, streaming may not be beneficial")
        
        return validation


class ProcessedVideoEncoder:
    """JSON encoder for ProcessedVideo objects."""
    
    @staticmethod
    def encode(processed_video: ProcessedVideo) -> Dict[str, Any]:
        """Encode ProcessedVideo to JSON-serializable dictionary."""
        encoded = asdict(processed_video)
        
        # Handle tensor serialization
        if isinstance(processed_video.frames, torch.Tensor):
            encoded["frames"] = {
                "type": "torch.Tensor",
                "shape": list(processed_video.frames.shape),
                "dtype": str(processed_video.frames.dtype),
                "data": processed_video.frames.cpu().numpy().tolist()
            }
        elif isinstance(processed_video.frames, np.ndarray):
            encoded["frames"] = {
                "type": "numpy.ndarray", 
                "shape": list(processed_video.frames.shape),
                "dtype": str(processed_video.frames.dtype),
                "data": processed_video.frames.tolist()
            }
        
        return encoded
    
    @staticmethod
    def decode(encoded_data: Dict[str, Any]) -> ProcessedVideo:
        """Decode JSON dictionary back to ProcessedVideo."""
        # Handle frames deserialization
        frames_data = encoded_data["frames"]
        if isinstance(frames_data, dict):
            if frames_data["type"] == "torch.Tensor":
                frames = torch.tensor(frames_data["data"])
                frames = frames.reshape(frames_data["shape"])
            elif frames_data["type"] == "numpy.ndarray":
                frames = np.array(frames_data["data"])
                frames = frames.reshape(frames_data["shape"])
            else:
                frames = frames_data
        else:
            frames = frames_data
        
        return ProcessedVideo(
            frames=frames,
            metadata=encoded_data["metadata"],
            timing=encoded_data["timing"],
            token_info=encoded_data.get("token_info"),
            quality_info=encoded_data.get("quality_info"),
            cache_info=encoded_data.get("cache_info")
        )


def format_simple(video_tensor: torch.Tensor, format_type: str = "standard") -> Any:
    """
    Simple function to format video tensor.
    
    Args:
        video_tensor: Video tensor to format
        format_type: Output format type
        
    Returns:
        Formatted output
    """
    from ..config import get_default_config
    
    handler = FormatHandler(get_default_config())
    
    # Create minimal metadata and timing
    metadata = {
        "num_frames": video_tensor.shape[0] if video_tensor is not None else 0,
        "height": video_tensor.shape[2] if video_tensor is not None else 0,
        "width": video_tensor.shape[3] if video_tensor is not None else 0,
    }
    timing = {"total_time": 0.0}
    
    return handler.format_output(video_tensor, metadata, timing, format_type) 