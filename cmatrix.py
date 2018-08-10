#This code is used to test the recogniser accuracy
#labelled_testset directory constains random images of faces cropped from the video feed and labelled manually
#The code runs all images through recogniser and creates confusion matrix to show accuracy
#the results are saved in csv format



# Import the required modules
import cv2, os
import numpy as np
import pandas as pd
from PIL import Image


# For face recognition we will the the LBPH Face Recognizer 
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("./utils/recog3.yml")
foo=0
y = [[foo for i in range(15)] for j in range(15)]
print(np.matrix(y))
path = './labelled_testset'
truepos=0
falsepos=0
# Append the images with the extension .sad into image_paths
image_ = [os.path.join(path, f) for f in os.listdir(path)]
for image in image_:
    predict_image_pil = Image.open(image).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    nbr_predicted, conf = rec.predict(predict_image)
    nbr_actual = int(os.path.split(image)[1].split(".")[0].replace("subject", ""))
    y[nbr_actual][nbr_predicted]= y[nbr_actual][nbr_predicted] + 1
    if nbr_predicted==nbr_actual:
    	truepos=truepos+1
    else:
    	falsepos=falsepos+1
    print("{} is Recognized as {}".format(nbr_actual, nbr_predicted))
    cv2.imshow("Recognizing Face", predict_image)
    cv2.waitKey(10)
print(np.matrix(y))
print("Accuracy is {0:.1f}%".format(truepos/(truepos+falsepos)*100))
#np.savetxt("new2.txt", np.matrix(y), fmt='%i')
df = pd.DataFrame(np.matrix(y))
df.to_csv("conf_matrix.csv")



