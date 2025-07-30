import yt_dlp
import os
import logging
import json
import subprocess
import ffmpeg
from datetime import datetime
from pathlib import Path

# Audio transcription imports
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tiktok_downloads.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhisperAudioTranscriber():
    def __init__(self, model_name="openai/whisper-large-v3-turbo"):
        # Configure the device for computation
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        # Load the model and processor
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_name)

            # Configure the pipeline for automatic speech recognition
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                return_timestamps=True,
                generate_kwargs={"max_new_tokens": 400},
                chunk_length_s=5,
                stride_length_s=(1, 1),
            )
            logger.info(f"🎤 Whisper model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe(self, audio_path: str) -> tuple:
        try:
            # Perform transcription with timestamps
            result = self.pipe(audio_path)
            transcription = result['text']
            timestamps = result['chunks']
            return transcription, timestamps
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None, None

class PyannoteAudioDiarizer():
    def __init__(self, model_name="pyannote/speaker-diarization-3.1", auth_token=None):
        try:
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=auth_token
            )
            logger.info("🎯 Pyannote diarization model loaded")
        except Exception as e:
            logger.error(f"Failed to load Pyannote model: {str(e)}")
            logger.warning("You may need to accept the user agreement at https://huggingface.co/pyannote/speaker-diarization")
            raise

    def diarize(self, audio_path: str) -> Annotation:
        try:
            diarization = self.pipeline(audio_path)
            return diarization
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            return None

class AudioAligner():
    def __init__(self):
        pass

    def align_transcription_with_diarization(self, transcription_segments, diarization):
        """
        Align Whisper transcription segments with Pyannote diarization results
        """
        try:
            aligned_segments = []
            
            for trans_segment in transcription_segments:
                start_time = trans_segment['timestamp'][0]
                end_time = trans_segment['timestamp'][1]
                text = trans_segment['text']
                
                # Find the speaker for this time segment
                speaker = self.find_speaker_at_time(diarization, start_time, end_time)
                
                aligned_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text.strip(),
                    'speaker': speaker
                })
            
            # Merge consecutive segments from the same speaker
            merged_segments = self.merge_consecutive_segments(aligned_segments)
            
            return merged_segments
        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            return []

    def find_speaker_at_time(self, diarization, start_time, end_time):
        """
        Find the most likely speaker for a given time segment
        """
        if not diarization:
            return "SPEAKER_UNKNOWN"
            
        mid_time = (start_time + end_time) / 2
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= mid_time <= segment.end:
                return speaker
        
        return "SPEAKER_UNKNOWN"

    def merge_consecutive_segments(self, segments):
        """
        Merge consecutive segments from the same speaker
        """
        if not segments:
            return []
        
        merged_segments = [segments[0]]
        
        for current_segment in segments[1:]:
            last_segment = merged_segments[-1]
            
            # Check if same speaker and segments are close in time (within 1 second)
            if (current_segment['speaker'] == last_segment['speaker'] and 
                current_segment['start'] - last_segment['end'] <= 1.0):
                # Merge segments
                last_segment['end'] = current_segment['end']
                last_segment['text'] += ' ' + current_segment['text']
            else:
                merged_segments.append(current_segment)
        
        return merged_segments

def extract_audio(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using FFmpeg
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file (WAV format)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if audio file already exists
        if os.path.exists(audio_path):
            logger.info(f"Audio file already exists: {audio_path}")
            return True
            
        # Extract audio using ffmpeg-python
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        
        logger.info(f"🎵 Audio extracted: {audio_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {str(e)}")
        return False

def transcribe_audio(audio_path: str, auth_token: str = None) -> dict:
    """
    Transcribe audio using Whisper + Pyannote pipeline
    
    Args:
        audio_path: Path to audio file
        auth_token: HuggingFace auth token for Pyannote (optional)
        
    Returns:
        Dictionary containing transcription results
    """
    try:
        logger.info(f"🎯 Starting transcription pipeline for: {audio_path}")
        
        # Initialize transcriber and diarizer
        transcriber = WhisperAudioTranscriber()
        
        # Perform transcription
        logger.info("🎤 Running Whisper transcription...")
        full_transcription, transcription_segments = transcriber.transcribe(audio_path)
        
        if not full_transcription:
            logger.error("Whisper transcription failed")
            return {}
            
        logger.info(f"✅ Transcription completed: {len(full_transcription)} characters")
        
        # Initialize diarization (optional, may fail without proper setup)
        diarization_result = None
        aligned_segments = []
        
        try:
            logger.info("🎯 Running Pyannote diarization...")
            diarizer = PyannoteAudioDiarizer(auth_token=auth_token)
            diarization_result = diarizer.diarize(audio_path)
            
            if diarization_result:
                # Align transcription with diarization
                logger.info("🔗 Aligning transcription with speaker segments...")
                aligner = AudioAligner()
                aligned_segments = aligner.align_transcription_with_diarization(
                    transcription_segments, diarization_result
                )
                logger.info(f"✅ Alignment completed: {len(aligned_segments)} segments")
            
        except Exception as e:
            logger.warning(f"Diarization failed (transcription will continue without speaker info): {str(e)}")
            # Convert transcription segments to simple format without speaker info
            aligned_segments = [{
                'start': seg['timestamp'][0],
                'end': seg['timestamp'][1],
                'text': seg['text'].strip(),
                'speaker': 'SPEAKER_UNKNOWN'
            } for seg in transcription_segments]
        
        # Prepare results
        results = {
            'transcription': {
                'full_text': full_transcription,
                'segments': aligned_segments,
                'total_segments': len(aligned_segments),
                'processing_date': datetime.now().isoformat()
            },
            'audio_info': {
                'file_path': audio_path,
                'file_exists': os.path.exists(audio_path),
                'file_size_mb': round(os.path.getsize(audio_path) / (1024*1024), 2) if os.path.exists(audio_path) else 0
            },
            'processing_settings': {
                'whisper_model': 'openai/whisper-large-v3-turbo',
                'pyannote_model': 'pyannote/speaker-diarization-3.1',
                'device_used': transcriber.device if transcriber else 'unknown',
                'diarization_enabled': diarization_result is not None
            }
        }
        
        logger.info("🎉 Audio transcription pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Transcription pipeline failed: {str(e)}")
        return {}

def save_video_description(video_info: dict, description_file: str, url: str, transcription_data: dict = None):
    """
    Save video description and metadata to a JSON file
    
    Args:
        video_info: Video information from yt-dlp
        description_file: Path to save the description file (.json)
        url: Original video URL
    """
    try:
        # Format upload date
        upload_date = video_info.get('upload_date', 'Unknown')
        formatted_date = upload_date
        if upload_date and upload_date != 'Unknown' and len(str(upload_date)) == 8:
            formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        
        # Format duration
        duration_seconds = video_info.get('duration', 0)
        duration_formatted = "00:00"
        if duration_seconds:
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            duration_formatted = f"{minutes:02d}:{seconds:02d}"
        
        # Create metadata dictionary
        metadata = {
            "title": video_info.get('title', 'Unknown Title'),
            "url": url,
            "uploader": video_info.get('uploader', 'Unknown'),
            "uploader_id": video_info.get('uploader_id', 'Unknown'),
            "upload_date": formatted_date,
            "duration_seconds": duration_seconds,
            "duration_formatted": duration_formatted,
            "view_count": video_info.get('view_count', 0),
            "like_count": video_info.get('like_count', 0),
            "comment_count": video_info.get('comment_count', 0),
            "description": video_info.get('description', ''),
            "webpage_url": video_info.get('webpage_url', url),
            "extractor": video_info.get('extractor', 'TikTok'),
            "video_id": video_info.get('id', ''),
            "download_date": datetime.now().isoformat(),
            
            # Audio transcription data
            "transcription": transcription_data if transcription_data else {},
            
            # Placeholder fields for future additions
            "tags": [],
            "notes": "",
            "categories": [],
            "custom_metadata": {}
        }
        
        # Save as JSON with pretty formatting
        with open(description_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📝 Description saved: {description_file}")
        
    except Exception as e:
        logger.error(f"Failed to save description file {description_file}: {str(e)}")

def activate_conda_environment():
    """
    Activate the conda environment for audio processing
    """
    try:
        # Try to activate the preferred conda environment
        env_name = "python12"  # Primary choice
        result = subprocess.run(
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {env_name}",
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            # Fallback to alternative environment
            env_name = "python11"
            result = subprocess.run(
                f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {env_name}",
                shell=True, capture_output=True, text=True
            )
        
        if result.returncode == 0:
            logger.info(f"🐍 Conda environment '{env_name}' activated")
        else:
            logger.warning(f"Could not activate conda environment. Proceeding with system Python.")
            
    except Exception as e:
        logger.warning(f"Conda activation failed: {str(e)}. Proceeding with system Python.")

def download_tiktok(url: str, output_dir: str = ".", video_number: int = None, total_videos: int = None, enable_transcription: bool = True):
    """
    Download a single TikTok video
    
    Args:
        url: TikTok video URL
        output_dir: Output directory for downloads
        video_number: Current video number for progress tracking
        total_videos: Total number of videos for progress tracking
    """
    try:
        # Extract user and video info from URL
        url_parts = url.split('/')
        user_name = url_parts[3][1:]  # Remove @ symbol
        video_id = url_parts[-1]
        
        progress_info = f"[{video_number}/{total_videos}]" if video_number and total_videos else ""
        logger.info(f"{progress_info} Processing video from @{user_name} (ID: {video_id})")
        
        # Create user directory
        save_path = f'{output_dir}/{user_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Created directory: {save_path}")
        
        # Use video ID in filename to avoid conflicts
        base_filename = f'{user_name}_{video_id}'
        output_filename = f'{save_path}/{base_filename}.mp4'
        description_filename = f'{save_path}/{base_filename}.json'
        audio_filename = f'{save_path}/{base_filename}.wav'
        
        # Skip if video, description, and audio files already exist
        if (os.path.exists(output_filename) and 
            os.path.exists(description_filename) and 
            (not enable_transcription or os.path.exists(audio_filename))):
            logger.info(f"Files already exist, skipping: {output_filename}")
            return True
        
        # If only video exists but not description, we'll re-download to get description
        if os.path.exists(output_filename) and not os.path.exists(description_filename):
            logger.info(f"Video exists but description missing, extracting info for: {output_filename}")
            # Extract info without downloading to create description file
            ydl_opts_info = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(url, download=False)
                save_video_description(info, description_filename, url)
                logger.info(f"📝 Description created for existing video")
                return True
        
        ydl_opts = {
            'outtmpl': output_filename,
            'format': 'mp4',
            'noplaylist': True,
            'quiet': True,  # Reduce yt-dlp output verbosity
            # Uncomment next line to remove watermarks if supported by your version
            # 'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown Title')
            
            # Process audio transcription if enabled
            transcription_data = {}
            if enable_transcription:
                try:
                    # Activate conda environment for audio processing
                    activate_conda_environment()
                    
                    # Extract audio from video
                    if extract_audio(output_filename, audio_filename):
                        # Transcribe audio
                        transcription_data = transcribe_audio(audio_filename)
                        
                        if transcription_data:
                            logger.info(f"📝 Transcription completed for: {title}")
                        else:
                            logger.warning(f"Transcription failed for: {title}")
                    else:
                        logger.warning(f"Audio extraction failed for: {title}")
                        
                except Exception as e:
                    logger.error(f"Audio processing failed for {title}: {str(e)}")
            
            # Save video description/metadata with transcription data
            save_video_description(info, description_filename, url, transcription_data)
            
            logger.info(f"✅ Successfully downloaded and processed: {title}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {str(e)}")
        return False

def load_urls_from_file(file_path: str) -> list:
    """
    Load TikTok URLs from a text file
    
    Args:
        file_path: Path to the text file containing URLs (one per line)
        
    Returns:
        List of URLs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]
        
        logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        return urls
        
    except FileNotFoundError:
        logger.error(f"URL file not found: {file_path}")
        logger.info("Please create a file with URLs (one per line) or check the file path.")
        return []
    except Exception as e:
        logger.error(f"Error reading URL file {file_path}: {str(e)}")
        return []

def download_multiple_tiktoks(urls: list, output_dir: str = ".", enable_transcription: bool = True):
    """
    Download multiple TikTok videos with progress tracking and error handling
    
    Args:
        urls: List of TikTok video URLs
        output_dir: Output directory for downloads
    """
    if not urls:
        logger.error("No URLs provided for download")
        return 0, 0
        
    total_videos = len(urls)
    successful_downloads = 0
    failed_downloads = 0
    
    logger.info(f"Starting download of {total_videos} TikTok videos")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    for i, url in enumerate(urls, 1):
        logger.info(f"\n{'='*50}")
        
        success = download_tiktok(url, output_dir, i, total_videos, enable_transcription)
        
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Progress update
        completion_percentage = (i / total_videos) * 100
        logger.info(f"Progress: {completion_percentage:.1f}% ({i}/{total_videos})")
    
    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info(f"DOWNLOAD SUMMARY:")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Success rate: {(successful_downloads/total_videos)*100:.1f}%")
    
    return successful_downloads, failed_downloads

if __name__ == "__main__":
    # File containing TikTok URLs (one per line)
    urls_file = "urls.txt"
    
    # Output directory
    output_directory = "/home/shuhang/code/video_process/tt_videos"
    
    # Load URLs from file
    urls = load_urls_from_file(urls_file)
    
    if urls:
        # Enable/disable transcription (set to False if you want to skip audio processing)
        enable_transcription = True
        
        logger.info(f"🎬 Starting TikTok download and processing...")
        logger.info(f"📁 Output directory: {output_directory}")
        logger.info(f"🎤 Audio transcription: {'Enabled' if enable_transcription else 'Disabled'}")
        
        if enable_transcription:
            logger.info("📋 Note: First run may take longer as models are downloaded")
            logger.info("📋 Note: Pyannote requires HuggingFace account and agreement acceptance")
        
        # Download all videos
        download_multiple_tiktoks(urls, output_directory, enable_transcription)
    else:
        logger.error("No URLs loaded. Please check your urls.txt file.")
