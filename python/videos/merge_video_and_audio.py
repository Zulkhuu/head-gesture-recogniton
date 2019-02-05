from moviepy.editor import *
from moviepy.video import *

# Load a movie and add the audio with it
video_clip = VideoFileClip("DakotaJohnson_facedet.mp4").set_audio( AudioFileClip("DakotaJohnson.mp3"))
video_clip.write_videofile("DakotaJohnson_facedet_with_audio.mp4") # default codec: 'libx264', 24 fps
