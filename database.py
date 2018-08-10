import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import inference_video_face
from inference_video_face import recog
import csv

data_students=[0]*81
attendance=[0]*81
result=[0]*81
	
cap= cv2.VideoCapture('sample.mp4'); 
frame_num=0
while(frame_num<=600):
	ret, image = cap.read()
	frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 
	print(frame_num)
	if (frame_num % 30 ==0):
		print("yes")
		image1=image
		#cv2.imshow('uhdu',image1)
		#cv2.waitKey(100)
		students=recog(image1)
		for x in range(80):
			data_students[x]=data_students[x]+students[x]
for z in range(80):
#10 is  the threshold
	if data_students[z]>2:
		attendance[z]=1
for y in range(80):

	if attendance[y]!=0:
		result[y]=y+3500
		print("%d is present"%(y+3500))
	
csvfile = "./output1.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    string="Lecture 1\n"
    writer.writerow([string])
    for val in result:
    	if val!=0:
    		writer.writerow([val])    



