from haar_blob import HaarCascadeBlobCapture
import cv2
import numpy as np
import urllib.request
URL = "http://192.168.57.133:8080/shot.jpg"
from random import random


while True:
    img_arr = np.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8
        )
    frame = cv2.imdecode(img_arr, -1)  
    capture = HaarCascadeBlobCapture()
    face, l_eye, r_eye = capture.process(frame, 60 , 60)
    cv2.imshow("frame", frame)
    if face is not None:
        cv2.imshow("face", face)
    if l_eye is not None:
        cv2.imshow("left eye", l_eye)
    if r_eye is not None:
        cv2.imshow("right eye", r_eye)
    key = cv2.waitKey(1)
    if key == 27:
        break;