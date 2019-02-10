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

import sys
import dlib
import cv2
import itertools
import progressbar
import numpy as np
import pandas as pd

predictor_path = '../lib/dlib-models/shape_predictor_5_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

filename = '2.yes_motion_resize'
vidin = cv2.VideoCapture('./videos/{:s}.mp4'.format(filename))
ret,frame = vidin.read()
fps = vidin.get(cv2.CAP_PROP_FPS)
frames = vidin.get(cv2.CAP_PROP_FRAME_COUNT)
results = []

print(' Video FPS rate is {}'.format(fps))
print(' {} total frames'.format(frames))
print(' Frame size : {}'.format(frame.shape))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidout = cv2.VideoWriter('./videos/{:s}-track-5landmarks.mp4'.format(filename),fourcc, 60.0, (frame.shape[1],frame.shape[0]))

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

def draw_landmarks(image, det, shape):
    color = (0,255,0)
    for i in range(shape.num_parts):
        cv2.circle(image, (shape.part(i).x, shape.part(i).y), 2, color, 2)

    return image

with progressbar.ProgressBar(max_value=frames) as bar:
    n = 0
    is_tracking = False
    tracker = cv2.TrackerKCF_create()
    while(vidin.isOpened()):
        ret, frame = vidin.read()
        if frame is None:
            break;
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start timer
        timer = cv2.getTickCount()

        if not is_tracking:
            dets = detector(rgb_image)
            for det in dets:
                bgr_frame = crosshairs(frame,det.left(),det.right(), det.top(),det.bottom())
                bbox = (det.left(),det.top(),det.right()-det.left(),det.bottom()-det.top())
                shape = predictor(rgb_image, det)
                bgr_frame = draw_landmarks(bgr_frame, det, shape)
                parts = [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]
                pts = list(itertools.chain(*parts))
                results.append([n, pts])
                print(results)
                is_tracking = tracker.init(frame, bbox)
                if is_tracking:
                    cv2.putText(frame, "Tracker init : ok", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
                else:
                    cv2.putText(frame, "Tracker init : failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
                #print('is_tracking:{}'.format(is_tracking))
                break
        else:
            # Update tracker
            track, bbox = tracker.update(frame)
            if track:
                # Tracking success
                frame = crosshairs(frame,int(bbox[0]),int(bbox[0]+bbox[2]),int(bbox[1]),int(bbox[1]+bbox[3]))
                det = dlib.rectangle(int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
                shape = predictor(rgb_image, det)
                frame = draw_landmarks(frame, det, shape)
                parts = [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]
                pts = list(itertools.chain(*parts))
                results.append([n, pts])
                cv2.putText(frame, "Tracking : ok", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
            else:
                tracker.clear()
                tracker = cv2.TrackerKCF_create()
                is_tracking = False
                cv2.putText(frame, "Tracking : lost", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        #for det in dets:
        #    bgr_frame[det.top():det.bottom(),det.left():det.right(),:] = frame[det.top():det.bottom(),det.left():det.right(),:]
        #    bgr_frame = crosshairs(bgr_frame,det.left(),det.right(), det.top(),det.bottom())
        #    break

        cv2.imshow('frame',frame)
        # write the flipped frame
        vidout.write(frame)
        n = n + 1
        bar.update(n)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Save score
df = pd.DataFrame(results,columns=['frame#','pts'])
df.to_csv('./csv/{:s}.csv'.format(filename)),

# Release everything if job is finished
vidin.release()
vidout.release()
cv2.destroyAllWindows()
