import sys
import dlib
import cv2
print(cv2.__version__)
import progressbar
import time

import face_alignment
import numpy as np

#predictor_path = '../weights/shape_predictor_68_face_landmarks.dat'
video_input = './videos/2.yes_motion_resize.mp4'
#video_input = './videos/Expression-Brow-Down.mp4'
video_output = './videos/2.yes_motion_resize_facedet.mp4'

# Run the 3D face alignment on a test image, without CUDA.
# https://github.com/1adrianb/face-alignment
# pip install face-alignment

#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True, face_detector='dlib')

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
vidout = cv2.VideoWriter(video_output,fourcc, fps/5, (frame.shape[1]*3,frame.shape[0]))

def rgb_to_gray(src):
     dist = src.copy()
     b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]
     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     dist[:,:,0] = gray
     dist[:,:,1] = gray
     dist[:,:,2] = gray
     return dist

def crosshairs(image,bbox):
    # Param
    color = (0,255,0)
    y_offset = 0
    # Line Property
    x_line_width = np.max([int(0.015*bbox[2]),1])
    y_line_width = np.max([int(0.015*bbox[3]),1])
    x_line_len   = np.max([int(0.15*bbox[2]),1])
    y_line_len   = np.max([int(0.15*bbox[3]),1])
    # Shift Y coordinate
    x_left   = int(bbox[0])
    x_right  = int(bbox[0] + bbox[2])
    y_top    = int(bbox[1] - y_offset)
    y_bottom = int(bbox[1] + bbox[3] - y_offset)
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

    cv2.polylines(image, [p1], False, (0,0,255), 2)
    cv2.polylines(image, [p2], False, (255,0,0), 2)
    cv2.polylines(image, [p3], False, (0,0,255), 2)
    cv2.polylines(image, [p4], False, (255,0,0), 2)
    cv2.polylines(image, [p5], False, color, 2)
    cv2.polylines(image, [p6], False, color, 2)
    cv2.polylines(image, [p7], False, (255,0,0), 2)
    cv2.polylines(image, [p8], False, (0,0,255), 2)

    for x,y,z in points:
        cv2.circle(image, (x, y), 2, color, -1)

    #x_left   = np.min(points[:,0]-20)
    #x_right  = np.max(points[:,0]+20)
    #y_top    = np.min(points[:,1]-20)
    #y_bottom = np.max(points[:,1]+20)
    #bbox = (x_left,y_top,x_right-x_left,y_bottom-y_top)
    return image #crosshairs(image,bbox)

with progressbar.ProgressBar(maxval=frames) as bar:
    n = 0
    is_tracking = False
    resize_scale = 0.5
    size = frame.shape[0], frame.shape[1]*3, 3
    while(vidin.isOpened()):
        n = n + 1
        ret, frame = vidin.read()
        bgr_image = cv2.resize(frame,None,fx=resize_scale,fy=resize_scale)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        bgr_frame = rgb_to_gray(frame)

        if n % 5 != 0:
            continue

        full_frame = np.zeros(size, dtype=np.uint8)
        full_frame[0:frame.shape[0],0:frame.shape[1],:] = bgr_frame
        # Start timer
        start_time = time.time()

        #detected_faces = fa.face_detector.detect_from_image(rgb_image[..., ::-1].copy())
        #for i, d in enumerate(detected_faces):
        #    center = [d[0] + 0.3 * (d[2] - d[0]) / 2.0, d[1] + 0.3 * (d[3] - d[1]) / 2.0]
        #    center[1] = center[1] + (d[3] - d[1]) * 0.12
        #    w = d[2] - d[0]
        #    h = d[3] - d[1]
        #    bbox = (center[0]-w/2,center[1]-h/2,center[0]+w/2,center[1]+h/2)
        #    frame = crosshairs(frame,bbox)
        points = fa.get_landmarks(rgb_image)
        if points:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            cv2.putText(full_frame, "Execution : {:.1f} [ms]".format(1000*elapsed_time), (470,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
            cv2.putText(full_frame, "Nod   : {:.1f}".format(0), (470,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
            cv2.putText(full_frame, "Shake : {:.1f}".format(0), (470,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
            #
            points = points[-1]/resize_scale
            x_left   = int(np.min(points[:,0]-12))
            x_right  = int(np.max(points[:,0]+12))
            y_top    = int(np.min(points[:,1]-12))
            y_bottom = int(np.max(points[:,1]+12))
            bbox = (x_left,y_top,x_right-x_left,y_bottom-y_top)
            full_frame[y_top:y_bottom,x_left:x_right,:] = frame[y_top:y_bottom,x_left:x_right,:]
            #
            for x,y,z in points:
                cv2.circle(full_frame, (x, y), 2, (255,255,255), -1)
            full_frame = crosshairs(full_frame,bbox)
            #
            p = points
            p[:,0] = points[:,0] + frame.shape[1]
            full_frame = draw_result(full_frame, p)
            #
            k = 0
            cv2.putText(full_frame, "  id     x       y       z     ",(frame.shape[1]*2+90,100+15*k), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50,50,50), 1, cv2.LINE_AA);
            for x,y,z in points[0:34,:]:
                k = k + 1
                cv2.putText(full_frame, "{:03d}  {:5.1f}   {:5.1f}   {:5.1f}".format(k,x,y,z), (frame.shape[1]*2+90,100+15*k), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50,50,50), 1, cv2.LINE_AA);
            cv2.putText(full_frame, "  id     x       y       z     ",(frame.shape[1]*2+280,100+15*(k-34)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50,50,50), 1, cv2.LINE_AA);
            for x,y,z in points[34:,:]:
                k = k + 1
                cv2.putText(full_frame, "{:03d}  {:5.1f}   {:5.1f}   {:5.1f}".format(k,x,y,z), (frame.shape[1]*2+280,100+15*(k-34)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50,50,50), 1, cv2.LINE_AA);


        # write the frame
        vidout.write(full_frame)

        #print('\r{:d}/{:d}'.format(n,int(frames)), end="")
        bar.update(n)

        cv2.namedWindow('image')
        cv2.imshow('image',full_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
vidin.release()
vidout.release()
cv2.destroyAllWindows()
#
