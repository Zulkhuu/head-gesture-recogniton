#!/usr/bin/python

import sys, os
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import itertools
import progressbar
import time
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def fddb_face_detect(image):
    """
    input: opencv image
    return: <left_x top_y width height detection_score>
    """
    dets = detector.detect_faces(image)
    results = []
    for det in dets:
        results.append('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(det['box'][0],det['box'][1],det['box'][2],det['box'][3],det['confidence']))
    return results

#Test FDDB dataset which is placed in ../dataset/FDDB
def test_fddb():
    database_path = '../dataset/FDDB'
    db_image_path = '{:s}/originalPics'.format(database_path)
    db_annotation_path = '{:s}/FDDB-folds'.format(database_path)
    n_annotation_file = 10
    for i in range(n_annotation_file):
        list_filename = '{:s}/FDDB-fold-{:02d}.txt'.format(db_annotation_path, i+1)
        output_filename = '{:s}/FDDB-fold-{:02d}-result.txt'.format(db_annotation_path, i+1)
        output_file = open(output_filename,mode='w')
        print('({:d}/{:d}) Detecting faces from image list in {:s} file'.format(i+1, n_annotation_file, list_filename))
        with progressbar.ProgressBar(maxval=len(open(list_filename).readlines())) as bar:
            n = 0
            with open(list_filename) as f:
                for image_filename in f:
                    bar.update(n)
                    n += 1
                    full_filename = '{:s}/{:s}.jpg'.format(db_image_path,image_filename.rstrip())
                    output_file.write(image_filename)
                    image_cv2 = cv2.imread(full_filename)
                    results = face_detect(image_cv2)
                    output_file.write('{:d}\n'.format(len(results)))
                    for result in results:
                        output_file.write('{:s}\n'.format(result))

test_fddb()
