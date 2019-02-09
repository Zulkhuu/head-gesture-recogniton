import sys
import dlib
import cv2
print(cv2.__version__)
import progressbar
import time

import face_alignment
import numpy as np

#predictor_path = '../weights/shape_predictor_68_face_landmarks.dat'
video_input = './videos/Expression-Brow-Down.mp4'
video_output = './videos/Expression-Brow-Down_facedet.mp4'

# Run the 3D face alignment on a test image, without CUDA.
# https://github.com/1adrianb/face-alignment
# pip install face-alignment

#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

#
vidin = cv2.VideoCapture(video_input)
ret,frame = vidin.read()
fps = vidin.get(cv2.CAP_PROP_FPS)
frames = vidin.get(cv2.CAP_PROP_FRAME_COUNT)

print(' Video FPS rate is {}'.format(fps))
print(' {} total frames'.format(frames))
print(' Frame size : {}'.format(frame.shape))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidout = cv2.VideoWriter(video_output,fourcc, fps/10, (frame.shape[1],frame.shape[0]))

def draw_result(image, points):
    color = (255,255,255)

    p1 = np.int32(points[ 0:9, :-1]).reshape(-1,1,2)
    p2 = np.int32(points[ 8:17,:-1]).reshape(-1,1,2)
    p3 = np.int32(points[17:22,:-1]).reshape(-1,1,2)
    p4 = np.int32(points[22:27,:-1]).reshape(-1,1,2)
    p5 = np.int32(points[27:31,:-1]).reshape(-1,1,2)
    p6 = np.int32(points[31:36,:-1]).reshape(-1,1,2)
    p7 = np.int32(points[48:60,:-1]).reshape(-1,1,2)
    p8 = np.int32(points[60:68,:-1]).reshape(-1,1,2)

    cv2.polylines(image, [p1], False, (0,0,255), 1)
    cv2.polylines(image, [p2], False, (255,0,0), 1)
    cv2.polylines(image, [p3], False, (0,0,255), 1)
    cv2.polylines(image, [p4], False, (255,0,0), 1)
    cv2.polylines(image, [p5], False, color, 1)
    cv2.polylines(image, [p6], False, color, 1)
    cv2.polylines(image, [p7], False, (255,0,0), 1)
    cv2.polylines(image, [p8], False, (0,0,255), 1)

    for x,y,z in points:
        cv2.circle(image, (x, y), 2, color, -1)

    return image

with progressbar.ProgressBar(maxval=frames) as bar:
    n = 0
    is_tracking = False
    while(vidin.isOpened()):
        n = n + 1
        ret, frame = vidin.read()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n % 10 != 0:
            continue
        # Start timer
        start_time = time.time()

        points = fa.get_landmarks(rgb_image)[-1]

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        cv2.putText(frame, "Execution time : {:.1f} ms".format(1000*elapsed_time), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        frame = draw_result(frame, points)

        # write the frame
        vidout.write(frame)
        #print('\r{:d}/{:d}'.format(n,int(frames)), end="")
        bar.update(n)

        cv2.namedWindow('image')
        cv2.imshow('image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
vidin.release()
vidout.release()
cv2.destroyAllWindows()
#
