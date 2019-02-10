from moviepy.editor import *
from moviepy.video import *

# Load a movie and add the audio with it
#video_clip = VideoFileClip("DakotaJohnson_facedet.mp4").set_audio( AudioFileClip("DakotaJohnson.mp3"))
video_clip = VideoFileClip("1.no_motion.mp4").resize((460,720))#.rotate(90)#.resize(0.25)#
video_clip.write_videofile("1.no_motion_resize.mp4") # default codec: 'libx264', 24 fps

video_clip = VideoFileClip("2.yes_motion.mp4").resize((460,720))#.rotate(90)#.resize(0.25)#
video_clip.write_videofile("2.yes_motion_resize.mp4") # default codec: 'libx264', 24 fps
