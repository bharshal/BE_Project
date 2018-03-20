#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
import pandas as pd
from PIL import Image


# For face recognition we will the the LBPH Face Recognizer 
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("./utils/recog4.yml")
foo=0
y = [[foo for i in range(21)] for j in range(21)]
print(np.matrix(y))
path = './test2'

# Append the images with the extension .sad into image_paths
image_ = [os.path.join(path, f) for f in os.listdir(path)]
for image in image_:
    predict_image_pil = Image.open(image).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    nbr_predicted, conf = rec.predict(predict_image)
    nbr_actual = int(os.path.split(image)[1].split(".")[0].replace("subject", ""))
    y[nbr_actual][nbr_predicted]= y[nbr_actual][nbr_predicted] + 1
    print("{} is Recognized as {}".format(nbr_actual, nbr_predicted))
    cv2.imshow("Recognizing Face", predict_image)
    cv2.waitKey(10)
print(np.matrix(y))
np.savetxt("new2.txt", np.matrix(y), fmt='%i')
df = pd.DataFrame(np.matrix(y))
df.to_csv("mat2.csv")



