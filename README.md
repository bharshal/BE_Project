# Automated Attendance System for Classroom using Face Recognition

This code is designed to create database of attendance for classroom from live video feed in real time. 

It doesn't require any input/intervention after code is run. 

It will take periodic frames from video feed and run face detection and recognition on those frames. 

As lecture is going on all students won't be recognised every time, hence it keeps record of each student in each frame and calculates number of times he/she was recognised. 

This also takes care of false positives and makes sure student is present throughout the lecture. 

At the end of the lecture it uses a threshold value to select which students should be marked present in the final database 


### Run `database.py` to get final output

This is the confidence when tested on set of 15 students.

|    | 0 | 1  | 2  | 3  | 4  | 5  | 6  | 7 | 8  | 9 | 10 | 11 | 12 | 13 | 14 |
|----|---|----|----|----|----|----|----|---|----|---|----|----|----|----|----|
| 0  | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0 | 0  | 0 | 0  | 0  | 0  | 0  | 0  |
| 1  | 0 | 16 | 0  | 3  | 0  | 1  | 0  | 2 | 0  | 0 | 0  | 2  | 0  | 0  | 1  |
| 2  | 0 | 0  | 11 | 1  | 0  | 0  | 0  | 1 | 0  | 0 | 0  | 0  | 0  | 1  | 1  |
| 3  | 0 | 0  | 1  | 28 | 0  | 0  | 0  | 4 | 0  | 0 | 2  | 2  | 0  | 0  | 0  |
| 4  | 0 | 2  | 0  | 1  | 15 | 1  | 1  | 0 | 1  | 4 | 2  | 1  | 0  | 0  | 0  |
| 5  | 0 | 0  | 0  | 1  | 0  | 14 | 0  | 0 | 0  | 1 | 0  | 0  | 0  | 0  | 0  |
| 6  | 0 | 0  | 0  | 0  | 0  | 1  | 12 | 0 | 0  | 0 | 0  | 0  | 0  | 0  | 0  |
| 7  | 0 | 0  | 0  | 1  | 2  | 0  | 0  | 6 | 0  | 0 | 0  | 2  | 0  | 0  | 1  |
| 8  | 0 | 0  | 0  | 1  | 0  | 0  | 0  | 0 | 12 | 0 | 3  | 0  | 0  | 0  | 0  |
| 9  | 0 | 0  | 0  | 0  | 0  | 2  | 1  | 0 | 0  | 8 | 0  | 0  | 0  | 0  | 1  |
| 10 | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0 | 0  | 0 | 13 | 0  | 0  | 0  | 0  |
| 11 | 0 | 1  | 2  | 0  | 0  | 0  | 0  | 1 | 2  | 0 | 1  | 23 | 0  | 1  | 0  |
| 12 | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0 | 0  | 0 | 0  | 0  | 6  | 0  | 0  |
| 13 | 0 | 0  | 2  | 0  | 0  | 0  | 0  | 0 | 0  | 0 | 0  | 1  | 0  | 4  | 0  |
| 14 | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0 | 0  | 0 | 0  | 0  | 0  | 0  | 16 |



You will need to paste these 2 files in the utils folder
    (1)shape_predictor_68_face_landmarks.dat
    (2)recog.yml
They are too large to be uploaded to github.

You can download them from following link and paste them in utils folder:
[Link](https://drive.google.com/open?id=1-O-2ad-HHmYQ6dJ13qHwzwvhlqOYX1T)

Code is under Apache 2.0 licence
The code uses Google's Tensorflow object detection api for face detection. Please refer to the license of tensorflow.
