import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2
import matplotlib.image as mpimg
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils_color as vis_util
from align_database.align_dlib import AlignDlib

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), './utils/shape_predictor_68_face_landmarks.dat'))

def _align_image(image, crop_dim):
	bb = align_dlib.getLargestFaceBoundingBox(image)
	aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
	return aligned

sys.path.append("..")
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("./utils/recog.yml")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './graph/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
	  (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
		
		
def recog(image):
	with detection_graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(graph=detection_graph, config=config) as sess:
			image_np=image
			#image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			#cv2.imshow('frame',image_np)
			#cv2.waitKey(100)
			# the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			# Actual detection.
			start_time = time.time()
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: 		 	image_np_expanded})


	elapsed_time = time.time() - start_time
	path="./crop"
	w,h=image.shape[:2]
	#print('inference time cost: {}'.format(elapsed_time))

	copy=image

	#image_path = [os.path.join(path, f) for f in os.listdir(path)]

	present_nos=[0]*81
	for i in range (0,100):
		#print(scores[0][1])
		if scores[0][i] > 0.4:
			a=(boxes[0][i][0])*w
			b=(boxes[0][i][1])*h
			c=(boxes[0][i][2])*w
			d=(boxes[0][i][3])*h
			a=int(a)
			b=int(b)
			c=int(c)
			d=int(d)
			if (a or c )<200:
				sub_face = copy[a-10:c+5,b-10:d+10]		
			else: 
				sub_face = copy[a-5:c+5,b-5:d+5]
			image1=np.array(sub_face,'uint8')
			new=cv2.resize(image1,(100,100))
			gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
			cl1 = clahe.apply(gray)
			predict_image = np.array(cl1, 'uint8')
			img2 = _align_image(cl1,80)
			if img2 is None:
				img3=predict_image
			elif img2.shape[:1] ==0 :
				img3=predict_image
			else:
				img3=img2
			#cv2.imshow('rec',img3)
			#cv2.waitKey(100)
			nbr_predicted, conf = rec.predict(img3)
			#print(conf)
			nbr_actual=int(i)
			#print("{} is Recognized as {}".format(nbr_actual, nbr_predicted))
			roll_no =nbr_predicted
			present_nos[roll_no]=1
			FaceFileName = path + "/%d"%i + '.jpeg'
			cv2.imwrite(FaceFileName, img3)
	return(present_nos)
	
if __name__ == "__main__":
	vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=2)
	#cv2.imshow('out',image)
	cv2.imwrite('./out1.jpeg',image)
#cv2.waitKey(10000)

	
