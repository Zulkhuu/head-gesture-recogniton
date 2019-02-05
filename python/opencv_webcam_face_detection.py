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

import sys
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        y_offset = 10

        w = det.right() - det.left()
        h = det.bottom() - det.top()
        x1 = det.left()
        y1 = det.top() - y_offset
        x2 = x1 + np.max([int(0.2*w),1])
        y2 = y1 + np.max([int(0.02*h),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x2 = x1 + np.max([int(0.02*h),1])
        y2 = y1 + np.max([int(0.2*w),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x1 = det.right()
        x2 = x1 - np.max([int(0.2*w),1])
        y2 = y1 + np.max([int(0.02*h),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x2 = x1 - np.max([int(0.02*h),1])
        y2 = y1 + np.max([int(0.2*w),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        y1 = det.bottom() - y_offset
        x2 = x1 - np.max([int(0.2*w),1])
        y2 = y1 - np.max([int(0.02*h),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x2 = x1 - np.max([int(0.02*h),1])
        y2 = y1 - np.max([int(0.2*w),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x1 = det.left()
        x2 = x1 + np.max([int(0.2*w),1])
        y2 = y1 - np.max([int(0.02*h),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)

        x2 = x1 + np.max([int(0.02*h),1])
        y2 = y1 - np.max([int(0.2*w),1])
        cv2.rectangle(img,(x1, y1), (x2, y2), color_green, -1)


        #print ('rect:{},{},{},{}'.format(x1,y1,x2,y2))

    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
