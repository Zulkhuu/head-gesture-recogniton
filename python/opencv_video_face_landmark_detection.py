#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in a webcam stream using OpenCV.
#   It is also meant to demonstrate that rgb images from Dlib can be used with opencv by just
#   swapping the Red and Blue channels.
#
#   You can run this program and see the detections from your webcam by executing the
#   following command:
#       ./opencv_face_detection.py
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy
#       pip install progressbar2

import sys
import dlib
import cv2
import numpy as np
#import progressbar

predictor_path = '../weights/shape_predictor_68_face_landmarks.dat'
video_input = './videos/DakotaJohnson.mp4'
video_output = './videos/DakotaJohnson_facedet.mp4'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vidin = cv2.VideoCapture(video_input)
ret,frame = vidin.read()
fps = vidin.get(cv2.CAP_PROP_FPS)
frames = vidin.get(cv2.CAP_PROP_FRAME_COUNT)

print(' Video FPS rate is {}'.format(fps))
print(' {} total frames'.format(frames))
print(' Frame size : {}'.format(frame.shape))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidout = cv2.VideoWriter(video_output,fourcc, 60.0, (frame.shape[1],frame.shape[0]))

def rgb_to_gray(src):
     dist = src.copy()
     b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]
     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     dist[:,:,0] = gray
     dist[:,:,1] = gray
     dist[:,:,2] = gray
     return dist

def crosshairs(image,x_left,x_right,y_top,y_bottom):
    # Param
    color = (0,255,0)
    y_offset = 0
    # Line Property
    w = x_right  - x_left
    h = y_bottom - y_top
    x_line_width = np.max([int(0.02*w),1])
    y_line_width = np.max([int(0.02*h),1])
    x_line_len   = np.max([ int(0.2*w),1])
    y_line_len   = np.max([ int(0.2*h),1])
    # Shift Y coordinate
    y_top = y_top - y_offset
    y_bottom = y_bottom - y_offset
    # Left Top Corner
    cv2.rectangle(image,(x_left, y_top), (x_left + x_line_len, y_top + y_line_width), color, -1)
    cv2.rectangle(image,(x_left, y_top), (x_left + x_line_width, y_top + y_line_len), color, -1)
    # Right Top Corner
    cv2.rectangle(image,(x_right - x_line_len, y_top), (x_right, y_top + y_line_width), color, -1)
    cv2.rectangle(image,(x_right - x_line_width, y_top), (x_right, y_top + y_line_len), color, -1)
    # Right Bottom Corner
    cv2.rectangle(image,(x_right - x_line_len, y_bottom - y_line_width), (x_right, y_bottom), color, -1)
    cv2.rectangle(image,(x_right - x_line_width, y_bottom - y_line_len), (x_right, y_bottom), color, -1)
    # Left Bottom Corner
    cv2.rectangle(image,(x_left, y_bottom - y_line_width), (x_left + x_line_len, y_bottom), color, -1)
    cv2.rectangle(image,(x_left, y_bottom - y_line_len), (x_left + x_line_width, y_bottom), color, -1)
    #
    return image

def draw_result(image, det, shape):
    image = crosshairs(image,det.left(),det.right(), det.top(),det.bottom())
    color = (0,255,0)
    for i in range(shape.num_parts):
        #image[shape.part(i).y, shape.part(i).x] = (0,255,0)
        #if(i != 0) and (i != 17) and (i != 22) and (i != 27) and (i != 31) and (i != 36) and (i != 42) and (i != 48) and (i != 60) and (i != 68):
            #cv2.line(image,
            #        (shape.part(i-1).x, shape.part(i-1).y),
            #        (shape.part(i).x, shape.part(i).y),color,1)
        cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, 1)

    return image


#with progressbar.ProgressBar(maxval=frames) as bar:
n = 0
while(vidin.isOpened()):
    ret, frame = vidin.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image, 1)
    bgr_frame = rgb_to_gray(rgb_image)

    for det in dets:
        bgr_frame[det.top():det.bottom(),det.left():det.right(),:] = frame[det.top():det.bottom(),det.left():det.right(),:]
        shape = predictor(rgb_image, det)
        bgr_frame = draw_result(bgr_frame, det, shape)
        break

    #cv2.imshow('frame',bgr_frame)
    # write the flipped frame
    vidout.write(bgr_frame)
    n = n + 1
    print('\r{:d}/{:d}'.format(n,int(frames)), end="")
    #bar.update(n)

    #cv2.namedWindow('image')
    #cv2.imshow('image',bgr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
vidin.release()
vidout.release()
cv2.destroyAllWindows()
