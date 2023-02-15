# for documentations see https://pytube.io/en/latest/user/quickstart.html#downloading-a-video

from pytube import YouTube

def check_video_streams(link, progressive=True, adaptive=False, only_audio=False, file_extension='mp4'):
    yt = YouTube(link)
    print(yt.title)
    print(yt.streams.filter(progressive = progressive, adaptive = adaptive, only_audio = only_audio, file_extension = file_extension))

def download_video(link, itag):
    yt = YouTube(link)
    stream = yt.streams.get_by_itag(itag)
    stream.download()

link = 'https://www.youtube.com/watch?v=TSSPDwAQLXs'
check_video_streams(link)
download_video(link, 22)