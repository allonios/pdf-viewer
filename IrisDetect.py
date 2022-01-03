from os import openpty
import cv2
import numpy as np
import urllib.request
from scipy import ndimage
import math
from pyautogui import screenshot

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
from numpy.lib.function_base import append
URL = "http://192.168.57.133:8080/shot.jpg"
from random import random

eye = cv2.CascadeClassifier('./haarcascade_eye.xml')
face = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

Kernal = np.ones((2, 2), np.uint8)  # Declare kernal for morphology

# Kenak = np.array([])
class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer_list = []

    def append(self, value):
        if len(self.buffer_list) >= self.size:
            self.buffer_list.pop(0)
        # print("adding", value)
        self.buffer_list.append(value)


class WinkDetectionBuffer(Buffer):
    def __init__(self, size):
        Buffer.__init__(self,size)
        self.wait_for_open = False

    def get_right_eye(self):
        return list(map(lambda state: state[0],self.buffer_list))
    
    def get_left_eye(self):
        return list(map(lambda state: state[1],self.buffer_list))
        
    def empty(self):
        self.buffer_list = []

    def check_wink(self):
        if self.wait_for_open and ((True, True) in self.buffer_list):
            self.wait_for_open = False

        if self.wait_for_open:
            self.empty()
            return False

        if len(self.buffer_list) < self.size:
            return False

        print(any(self.get_left_eye()))
        if (
            (any(self.get_left_eye()) and (not any(self.get_right_eye()))) or 
            (any(self.get_right_eye()) and (not any(self.get_left_eye())))
        ): 
            print("Winkings")
            self.empty()
            self.wait_for_open = True
            return True

class EyeTrackingBuffer(Buffer):
    def get_buffer_ys(self):
        return list(map(lambda coord: coord[0] - coord[1], self.buffer_list))        

    def check_orientation(self):
        ...

        # if len(self.buffer_list) < self.size:
        #     print("incomplete")
        #     return
        ys = self.get_buffer_ys()
        avg = sum(ys) / self.size
        # print("avg: ", avg)
        # print("buffer: ", self.buffer_list)
        # if avg > 100:
        #     print("down")        

        # elif avg < 90:
        #     print("up")

        # else:
        #     print("strait")

eye_tracking_buffers = [EyeTrackingBuffer(5), EyeTrackingBuffer(5)]
wink_detection_buffer = WinkDetectionBuffer(3)

def which_eye(eye_x, eye_w, face_w):
    if eye_x + 0.5 * eye_w > (face_w / 2):
        return 0
    else:
        return 1



def detect_eye(window):
    for index, buffer in enumerate(eye_tracking_buffers):
        # print(f"buffer index {index}" , sep=": ")
        buffer.check_orientation()

    img_arr = np.array(
    bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.flip(frame, 1)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    cv2.imshow("gray", lab[:,:,0])
    gray = cv2.equalizeHist(lab[:,:,0])
    detect_face = face.detectMultiScale(gray, 1.3, 5)
    detect_eye = eye.detectMultiScale(gray, 1.3, 5)
    for(face_x, face_y, face_w, face_h) in detect_face:
        if wink_detection_buffer.check_wink():
            shot = screenshot()
            shot.save(r"/home/bilalx/Pictures/screenshot.jpg")
            
        img2 = gray[face_y:face_y+face_h, face_x:face_x+face_w]
        detect_eye = eye.detectMultiScale(img2, 1.3, 5)
        left_eye_detected = False
        right_eye_detected = False
        for index, (eye_x, eye_y, eye_w, eye_h) in enumerate(detect_eye):
            eye_index = which_eye(eye_x, eye_w, face_w)
            if eye_index == 0:
                print("left")
                left_eye_detected = True
            if eye_index == 1:
                print("right")
                right_eye_detected = True
            eye_tracking_buffers[eye_index].append((eye_y, eye_h))
            if index >= 2:
                continue
            if eye_y > 0.3 * face_h:
                continue
            if eye_h > 0.4 * face_h:
                continue

            eye1 = gray[face_y+eye_y:face_y+eye_y +
                        eye_h, face_x+eye_x:face_x+eye_x+eye_w]
            
            # width, height = eye1.shape
            # eye1 = eye1[int(0.3 * height):height, :]

            # eye1 = cv2.equalizeHist(eye1)
            # cv2.imshow("hist_equalized", hist_equalized)
            # cv2.imshow("eye1", eye1)
            eye1 = cv2.GaussianBlur(eye1, (3,3), 2)
            histRange = (0, 256)
            hist = cv2.calcHist([eye1],[0],None,[256],histRange, accumulate=False)
            # print(hist)
            eye_shape = eye1.shape[0] * eye1.shape[1]
            # print(eye_shape)
            hist = np.divide(hist, eye_shape) 
            # cv2.normalize(hist,hist,0,1,cv2.NORM_RELATIVE)
            cdf = np.add.accumulate(hist)
            # print(cdf)



            mask = np.where(
                np.less_equal(cdf[eye1], 0.05),
                255,
                0
            ).astype("uint8")

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_ERODE, Kernal, iterations=1)
            
            cv2.imshow("mask", mask)
            # for h in hist:

            not_mask = cv2.bitwise_not(mask)
            im = cv2.bitwise_or(not_mask, eye1)
            
            min_index = np.unravel_index(np.argmin(im), im.shape)
            window = eye1[min_index[0] - 5: min_index[0] + 5, min_index[1] - 5: min_index[1] + 5]
            threshold = np.average(window)
            iris = eye1[min_index[0] - 7: min_index[0] + 7, min_index[1] - 7: min_index[1] + 7]
            eroded_iris = cv2.morphologyEx(
                    mask, cv2.MORPH_ERODE, Kernal, iterations=1)

            # cv2.imshow("iris", iris)

            ret, binary = cv2.threshold(iris, threshold, 255, cv2.THRESH_BINARY_INV)
            new_iris = cv2.bitwise_and(iris, iris, mask=binary)

            if new_iris is None:
                continue
            # print(new_iris)
            cv2.imshow("new_iris", new_iris)
            center_of_mass = ndimage.measurements.center_of_mass(new_iris)
            # print("center_of_mass",center_of_mass)
            if math.isnan(center_of_mass[0]) or math.isnan(center_of_mass[1]):
                continue
            real_eye1_center = (
                int(center_of_mass[0] + min_index[0] - 7), int(center_of_mass[1] + min_index[1] - 7) 
            )


            eye2 = eye1.copy()
            cv2.circle(eye2,np.array(real_eye1_center), 2, (255,0,0), 2)
            # cv2.circle(eye1,min_index, 2, (255,0,0), 2)
            # cv2.imshow("algorithm", eye1)
            cv2.imshow("algorithm2", eye2)

            contours, hierarchy = cv2.findContours(iris, cv2.RETR_TREE,  # Find contours
                                                   cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(eye1, contours, 0, (255, 0, 0), 3)
            cv2.imshow("eye1",eye1)
            cv2.circle(img2, (int( eye_x + 0.5 * eye_w), int(eye_y + 0.5* eye_w)), 1, (255, 0, 0), 2)
            if len(contours) != 0:
                cnt = contours[0]
                M1 = cv2.moments(cnt)

                Cx1 = int(M1['m10'] / M1['m00'])  # Find center of the contour
                Cy1 = int(M1['m01'] / M1['m00'])
                # Number of pixels we cropped from the image
                iris_center = (int(Cx1+(min_index[0] - 7)+face_x+eye_x), int(Cy1 + (min_index[1] - 7)+face_y +
                                                      eye_y))  # Center coordinates
                eye_center = (int(eye_x + (0.5 * eye_w) +face_x), int(eye_y + (0.5 * eye_h) +face_y))  # Center coordinates
                eye_center_according_to_face = face_h / int(eye_y + (0.5 * eye_h))  # Center coordinates

                print("eye_center_according_to_face: ", eye_center_according_to_face)

                cv2.circle(frame, iris_center, 2, (255,0,0), 2)
                
                # print(iris_center[1] - eye_center[1])
                # if iris_center[1] - eye_center[1] > 2:
                #     print("up")
                # elif iris_center[1] - eye_center[1] < -2:
                #     print("down")
                # else:
                #     print("strait")
                # print(Cx1,Cy1)

        cv2.imshow("img2",img2)
        cv2.imshow('Frame Image', frame)  # Show original Image
        if right_eye_detected or left_eye_detected:
            print(wink_detection_buffer.buffer_list)
            wink_detection_buffer.append((right_eye_detected, left_eye_detected))

# while 1:
#     for index, buffer in enumerate(buffers):
#         print(f"buffer index {index}" , sep=": ")
#         buffer.check_orientation()

    #     img_arr = np.array(
    #     bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    #     frame = cv2.imdecode(img_arr, -1)
#     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     frame = cv2.flip(frame, 1)
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     cv2.imshow("gray", lab[:,:,0])
#     gray = cv2.equalizeHist(lab[:,:,0])
#     detect_face = face.detectMultiScale(gray, 1.3, 5)
#     detect_eye = eye.detectMultiScale(gray, 1.3, 5)
#     for(face_x, face_y, face_w, face_h) in detect_face:
#         img2 = gray[face_y:face_y+face_h, face_x:face_x+face_w]
#         detect_eye = eye.detectMultiScale(img2, 1.3, 5)
#         for index, (eye_x, eye_y, eye_w, eye_h) in enumerate(detect_eye):
#             eye_index = which_eye(eye_x, eye_w, face_w)
#             buffers[eye_index].append((eye_y, eye_h))
#             if index >= 2:
#                 continue
#             if eye_y > 0.3 * face_h:
#                 continue
#             if eye_h > 0.4 * face_h:
#                 continue

#             eye1 = gray[face_y+eye_y:face_y+eye_y +
#                         eye_h, face_x+eye_x:face_x+eye_x+eye_w]

#             # eye1 = cv2.equalizeHist(eye1)

#             # cv2.imshow("hist_equalized", hist_equalized)
#             # cv2.imshow("eye1", eye1)

#             eye1 = cv2.medianBlur(eye1, 3, 2)
#             blur = cv2.GaussianBlur(eye1,(3,3),2)
#             blur_gaussian_5 = cv2.GaussianBlur(eye1,(5,5),2)

            
#             equalized = cv2.equalizeHist(eye1)

#             grad_x = cv2.Sobel(equalized, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#             grad_y = cv2.Sobel(equalized, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#             abs_grad_x = cv2.convertScaleAbs(grad_x)
#             abs_grad_y = cv2.convertScaleAbs(grad_y)
#             grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            
#             # eye2 = cv2.GaussianBlur(eye1, (9,9), 2)
#             circles = cv2.HoughCircles(grad, cv2.HOUGH_GRADIENT, 1, 200,
#                 param1=50, param2=25, 
#                 minRadius=1, maxRadius=30)
#             if circles is not None:
#                 circles = np.uint16(np.around(circles))
#                 for i in circles[0, :]:
#                     center = (i[0], i[1])
#                     # circle center
#                     cv2.circle(grad, center, 1, (0, 100, 100), 1)
#                     # circle outline
#                     radius = i[2]
#                     cv2.circle(grad, center, radius, (255, 0, 255), 1)
#             cv2.imshow("eye2", grad)

            
#             # cv2.imshow("blured_median", median_blur)
#             cv2.imshow("blured_median_5", cv2.medianBlur(eye1, 5, 2))
#             cv2.imshow("blured_gaussian", eye1)
#             cv2.imshow("blured_gaussian_5", blur_gaussian_5)
#             ret3, otsu = cv2.threshold(eye1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)           

#             ret3, otsu_5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#             ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
#             adaptive_gaussian = cv2.adaptiveThreshold(eye1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                 cv2.THRESH_BINARY,11,2)
#             adaptive_mean = cv2.adaptiveThreshold(eye1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2) 

   

#             cv2.imshow("otsu", otsu)
#             cv2.imshow("otsu_5", otsu_5)
#             cv2.imshow("binary", binary)
#             cv2.imshow("adaptive_gaussian", adaptive_gaussian)
#             cv2.imshow("adaptive_mean", adaptive_mean)
#             cv2.imshow("adaptive_gaussian", adaptive_gaussian)
#             cv2.imshow("sobel", grad)

#             otsu_5_eroded = cv2.bitwise_not(otsu_5)
#             cv2.imshow("otsu_5_not", otsu_5_eroded)
#             otsu_5_eroded = cv2.morphologyEx(
#                 otsu_5_eroded, cv2.MORPH_ERODE, Kernal, iterations=20)
            
#             cv2.imshow("otsu_5_eroded", otsu_5_eroded)
#             binary=cv2.bitwise_not(adaptive_mean)

#             width, height = binary.shape

#             binary = binary[int(0.3 * height):height, :]
#             cv2.imshow("cropped", binary)
#             opening = cv2.morphologyEx(
#                 binary, cv2.MORPH_OPEN, Kernal, iterations=2)
            
#             cv2.imshow("opened", opening)
            
#             dilate = cv2.morphologyEx(
#                 opening, cv2.MORPH_DILATE, Kernal, iterations=4)  # Dilate Morphology

#             cv2.imshow("dilated", dilate)

#             contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE,  # Find contours
#                                                    cv2.CHAIN_APPROX_NONE)
#             cv2.drawContours(eye1, contours, 0, (255, 0, 0), 3)
#             cv2.circle(img2, (int(eye_x + 0.5 * eye_w), int(eye_y + 0.5* eye_w)), 1, (255, 0, 0), 2)
#             if len(contours) != 0:
#                 cnt = contours[0]
#                 M1 = cv2.moments(cnt)

#                 Cx1 = int(M1['m10'] / M1['m00'])  # Find center of the contour
#                 Cy1 = int(M1['m01'] / M1['m00'])
#                 # Number of pixels we cropped from the image
#                 croppedImagePixelLength = int(0.3*height)
#                 iris_center = (int(Cx1+face_x+eye_x), int(Cy1+face_y +
#                                                       eye_y + croppedImagePixelLength))  # Center coordinates
#                 eye_center = (int(eye_x + (0.5 * eye_w) +face_x), int(eye_y + (0.5 * eye_h) +face_y
#                                                       ))  # Center coordinates
#                 # print(iris_center[1] - eye_center[1])
#                 # if iris_center[1] - eye_center[1] > 2:
#                 #     print("up")
#                 # elif iris_center[1] - eye_center[1] < -2:
#                 #     print("down")
#                 # else:
#                 #     print("strait")
#                 # print(Cx1,Cy1)

#                 cv2.circle(frame, iris_center, 2, (0, 255, 0), 2)
#                 cv2.circle(frame, eye_center, 2, (0, 0, 255), 2)
#         cv2.imshow("img2",img2)
            

#     if not frame.any():  # If frame is not read then exit
#         break
#     if cv2.waitKey(1) == ord('s'):  # While loop exit condition
#         break
#     cv2.imshow('Frame Image', frame)  # Show original Image

# # cap.release()
# cv2.destroyAllWindows()
