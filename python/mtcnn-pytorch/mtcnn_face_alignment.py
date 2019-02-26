#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy
#       pip install progressbar2

import sys, os
import cv2
import numpy as np
import pandas as pd
import itertools
import progressbar
import time

from mtcnn.mtcnn import MTCNN

import_path = '../videos'
export_path = '../videos/result/'

program_name = 'mtcnn_pytorch_face_alignment'
input_filename = 'Expression-Brow-Down'

video_input = '{:s}/{:s}.mp4'.format(import_path,input_filename)
video_output = '{:s}/{:s}_{:s}.mp4'.format(export_path,input_filename,program_name)

if not os.path.exists(export_path):
    os.makedirs(export_path)

# Detector
detector = MTCNN()

# Video input
vidin = cv2.VideoCapture(video_input)
ret,frame = vidin.read()
fps = vidin.get(cv2.CAP_PROP_FPS)
frames = vidin.get(cv2.CAP_PROP_FRAME_COUNT)
results = {}

print(' Video fps rate is {}'.format(fps))
print(' {} total frames'.format(frames))
print(' Frame size : {}'.format(frame.shape))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidout = cv2.VideoWriter(video_output,fourcc, fps, (frame.shape[1],frame.shape[0]))

def rgb_to_gray(src):
     dist = src.copy()
     b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]
     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     dist[:,:,0] = gray
     dist[:,:,1] = gray
     dist[:,:,2] = gray
     return dist

def draw_crosshairs(image,x_left,x_right,y_top,y_bottom):
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

def draw_result(image, det):
    bbox = det['box']
    pts = det['keypoints']
    confidence = det['confidence']
    image = draw_crosshairs(image, bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3])
    color = (0,255,0)
    for key, pt in pts.items():
        cv2.circle(image, (pt[0], pt[1]), 2, color, 2)
    return image

with progressbar.ProgressBar(maxval=frames) as bar:
    n = 0
    while(vidin.isOpened()):
        ret, frame = vidin.read()
        if ret is True:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bgr_frame = rgb_to_gray(rgb_image)

            # Start timer
            start_time = time.time()

            dets = detector.detect_faces(rgb_image)
            for det in dets:
                bgr_frame = draw_result(bgr_frame, det)
                #results[n] = det
                #break

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            cv2.putText(bgr_frame, "Video FPS rate is {}".format(fps),                        (10,20),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);
            cv2.putText(bgr_frame, "{:d} total frames".format(int(frames)),                   (10,40),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);
            cv2.putText(bgr_frame, "Frame size : {}".format(frame.shape),                     (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);
            cv2.putText(bgr_frame, "Execution  : {:04d} [ms]".format(int(1000*elapsed_time)), (10,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);
            cv2.putText(bgr_frame, "Nod   : {:01d}".format(0),                                (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);
            cv2.putText(bgr_frame, "Shake : {:01d}".format(0),                                (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,0), 1, cv2.LINE_AA);

            # write the flipped frame
            bar.update(n)
            n += 1
            vidout.write(bgr_frame)

            #cv2.imshow('image',bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Save score
#df = pd.DataFrame.from_dict(results, orient='index')
#df.to_csv('./csv/{:s}.csv'.format(output_filename))

# Release everything if job is finished
vidin.release()
vidout.release()
cv2.destroyAllWindows()
