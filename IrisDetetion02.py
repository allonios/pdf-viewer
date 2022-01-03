from typing import get_args
import cv2
import sys
import numpy as np

# TO DO: Make this a function. Make main.py after pdf gets sorted out, then send frame to this function and return eye coordinates.
eyeCascade = cv2.CascadeClassifier('./haarcascade_eye_2.xml')
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)


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

    def __init__(self, size):
        Buffer.__init__(self,size)
        self.initial_value = False
        self.min_value = 0
        self.max_value = 0

    def append(self, value):
        # print("value: ", value)
        # if self.initial_value and self.value > self.max_value:
        #     print("DOWN") 
        # elif self.initial_value and self.value < self.min_value:
        #     print("UP")
        # else: 
        #     print("STRAIT")
        save_initial = False
        if (not self.initial_value) and len(self.buffer_list) == self.size - 1:
            self.initial_value = True
            save_initial = True
        Buffer.append(self,value)
        if save_initial:
            s = sorted(self.buffer_list)
            self.min_value = s[int(self.size/2)] + 0.15
            self.max_value = s[int(self.size/2)] + 0.15
        print(self.min_value)
        print(self.max_value)
        self.initial_value = False

    def get_avg(self):
        new_list = self.buffer_list[int(self.size/2):]
        avg = sum(new_list) / self.size/2
        return avg

    def check_orientation(self):
        if not self.initial_value:
            return 0
        avg = self.get_avg()
        if avg > self.max_value:
            return 1
        if avg > self.max_value:
            return -1
        return 0
        

buffers = [EyeTrackingBuffer(10), EyeTrackingBuffer(10)]

def which_eye(eye_x, eye_w, face_w):
    if eye_x + 0.5 * eye_w > (face_w / 2):
        print("right")
        return 0
    else:
        print("left")
        return 1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    area=[]
    fc=[0,0,500,500]
    xe=0
    ye=0
    #print type(faces)
    for (x,y,w,h) in faces:
        area.append(([x,y,w,h],h*w))
        cv2.rectangle(frame, (0,0), (1280,720), (0, 255, 0), 2)
        cv2.rectangle(frame, (0,0), (900,900), (0, 255, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        # cv2.rectangle(frame, (0,y_wind-1100), (200,y_wind-900), (0, 255, 0), 2)
        #f2=frame[y:y+h,x:x+w]

    if area:
        result0 = buffers[0].check_orientation()
        result1 = buffers[1].check_orientation()
        # if result0 == 1:
        #     print("0 => UP")
        # elif result0 == -1:
        #     print("0 => DOWN")
        # else:
        #     print("0 => STRAIT")

        # if result1 == 1:
        #     print("1 => UP")
        # elif result1 == -1:
        #     print("1 => DOWN")
        # else:
        #     print("1 => STRAIT")


        face_tracked=sorted(area,key=lambda x:x[1],reverse=True)[0][0]
        face_gray=gray[face_tracked[1]:face_tracked[1]+face_tracked[3],face_tracked[0]:face_tracked[0]+face_tracked[2]]
        eyes = eyeCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in eyes:
            eye_index = which_eye(x, w, face_tracked[2])
            eye_center_according_to_face = face_tracked[3] / int(y + (0.5 * h))  # Center coordinates
            print("eye_index: ", eye_index, )
            print("eye_center_according_to_face: ", eye_center_according_to_face)
            if eye_center_according_to_face > 2.7:
                print("DOWN")
            if eye_center_according_to_face < 2.55:
                print("UP")
            
            buffers[eye_index].append(eye_center_according_to_face)



            xe=x+w/2
            ye=y+h/2
            cv2.circle(frame, (int(xe+face_tracked[0]),int(ye+face_tracked[1])),2, (0,255,0),2)
            if not len(eyes)==0:
                xe=xe/len(eyes)+face_tracked[0]
                ye=ye/len(eyes)+face_tracked[1]
                #cv2.circle(frame, (xe,ye),2, (0,255,0),8)
        xf=face_tracked[0]+face_tracked[2]/2
        yf=face_tracked[1]+face_tracked[3]/3
        #print xf,yf
    # Display the resulting frame
    cv2.imshow('frame', frame)
    #f3=frame[fc[0]:fc[0]+fc[2],fc[1]:fc[1]+fc[3]]
    #cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    #cv2.imshow('frame2', f3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows
