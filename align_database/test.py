import argparse
import glob
import logging
import multiprocessing as mp
import os
import time

import cv2

from align_dlib import AlignDlib

logger = logging.getLogger(__name__)
path='./test'
path2='./test2'

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    return aligned
    
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
images = []
for image_path in image_paths:
	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
	cl1 = clahe.apply(gray)
	FaceFileName = (os.path.split(image_path)[1].split(".")[0]+'.'+os.path.split(image_path)[1].split(".")[1]) + '.jpeg'	
	#cv2.imshow('orig',img)  
	img2 = _align_image(cl1,100)
	if img2 is None:
		#cv2.imshow('frame',img)
		cl2=cv2.resize(cl1,(100,100))
		cv2.imwrite(os.path.join(path2,FaceFileName),cl2)
	elif img2.shape[:1] ==0 :
		#cv2.imshow('frame',img)
		cl2=cv2.resize(cl1,(100,100))
		cv2.imwrite(os.path.join(path2,FaceFileName),cl2)
	else:
		#cv2.imshow('frame',img2)
		cv2.imwrite(os.path.join(path2,FaceFileName),img2)
	cv2.waitKey(100)
	cv2.destroyAllWindows()
cv2.waitKey(50)
cv2.destroyAllWindows()
	
