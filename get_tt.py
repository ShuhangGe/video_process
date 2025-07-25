import yt_dlp
import os
def download_tiktok(url: str, output_dir: str = "."):
    user_name = url.split('/')[3][1:]
    video_name = url.split('/')[-1]
    save_path = f'{output_dir}/{user_name}'
    print(user_name)
    print(video_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ydl_opts = {
        'outtmpl': f'{save_path}/{user_name}.mp4',
        'format': 'mp4',
        'noplaylist': True,
        # Uncomment next line to remove watermarks if supported by your version
        # 'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("Downloaded:", info.get('title'))

if __name__ == "__main__":
    video_url = "https://www.tiktok.com/@aka0429/video/7384054578074963205"
    download_tiktok(video_url, output_dir="/Users/shuhangge/Desktop/my_projects/video_process/videos")
