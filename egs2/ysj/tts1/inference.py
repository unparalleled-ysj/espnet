import os

webm_dir = "/Users/ysj/资料/音乐/webm"
mp3_dir = "/Users/ysj/资料/音乐/mp3"
for name in os.listdir(webm_dir):
    name = name.replace(' ', '\ ')
    webm_path = os.path.join(webm_dir, name)
    mp3_path = os.path.join(mp3_dir, name.split('.')[0] + '.mp3')
    if not os.path.isfile(mp3_path.replace('\ ', ' ')):
        print(mp3_path)
#         os.system(f"ffmpeg -i {webm_path} -acodec libmp3lame {mp3_path}")
