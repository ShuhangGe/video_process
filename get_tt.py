import yt_dlp
import os
import logging
import json
from datetime import datetime

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

def save_video_description(video_info: dict, description_file: str, url: str):
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

def download_tiktok(url: str, output_dir: str = ".", video_number: int = None, total_videos: int = None):
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
        
        # Skip if both video and description files already exist
        if os.path.exists(output_filename) and os.path.exists(description_filename):
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
            
            # Save video description/metadata to txt file
            save_video_description(info, description_filename, url)
            
            logger.info(f"✅ Successfully downloaded: {title}")
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

def download_multiple_tiktoks(urls: list, output_dir: str = "."):
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
        
        success = download_tiktok(url, output_dir, i, total_videos)
        
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
        # Download all videos
        download_multiple_tiktoks(urls, output_directory)
    else:
        logger.error("No URLs loaded. Please check your urls.txt file.")
