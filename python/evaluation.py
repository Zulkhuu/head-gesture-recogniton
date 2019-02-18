#!/usr/bin/python

import sys, os
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import itertools
import progressbar
import time
#from mtcnn.mtcnn import MTCNN

#detector = MTCNN()

def test_pose():
    database_path = '../dataset/FP04DB'
    for x in os.walk(database_path):
        sub_dir = x[0]
        print(sub_dir)

#Test FDDB dataset which is placed in ../dataset/FDDB
def test_det():
    database_path = '../dataset/FDDB'
    db_image_path = '{:s}/originalPics'.format(database_path)
    db_annotation_path = '{:s}/FDDB-folds'.format(database_path)
    n_annotation_file = 10
    for i in range(n_annotation_file):
        list_filename = '{:s}/FDDB-fold-{:02d}.txt'.format(db_annotation_path, i+1)
        output_filename = '{:s}/FDDB-fold-{:02d}-result.txt'.format(db_annotation_path, i+1)
        print(list_filename)
        with open(list_filename) as f:
            for image_filename in f:
                print('{:s}/{:s}.jpg'.format(db_image_path,image_filename))



test_det()
