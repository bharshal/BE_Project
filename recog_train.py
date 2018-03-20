import numpy as np
import cv2
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images(path):
	image_ = [os.path.join(path, f) for f in os.listdir(path)]
	images2=[]
	labels=[]
	for	image in image_:
		image_pil1 = Image.open(image).convert('L')
		image1=np.array(image_pil1,'uint8')
		#new=cv2.resize(image1,(100,100))
		nbr = int(os.path.split(image)[1].split(".")[0].replace("subject", ""))
		images2.append(image1)
		labels.append(nbr)
		cv2.imshow("Adding faces to traning set...", image1)
		#cv2.waitkey(50)
	return images2,labels
# Path to the Dataset
path = './align_database/aligned'
# Call the get_images_and_labels function and get the face images and the corresponding labels
images2,labels=get_images(path)
cv2.destroyAllWindows()
# Perform the tranining
recognizer.train(images2, np.array(labels))
recognizer.save('./utils/recog4.yml')
