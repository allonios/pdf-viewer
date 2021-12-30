import cv2
import numpy as np


##Import xml files for face and eye detection##
eye = cv2.CascadeClassifier('./haarcascade_eye.xml')
face = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

Kernal = np.ones((3, 3), np.uint8)  # Declare kernal for morphology

cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


while 1:
    ret, frame = cap.read()  # Read image frame
    # Flip the image in case your camera capures inverted images
    frame = cv2.flip(frame, +1)
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = face.detectMultiScale(gray, 1.2, 1)  # Detect Face
    detect_eye = eye.detectMultiScale(gray, 1.2, 1)

    for(face_x, face_y, face_z, face_h) in detect_face:
        img2 = gray[face_y:face_y+face_h, face_x:face_x+face_z]
        # cv2.rectangle(frame, (face_x, face_y), (face_x+face_z, face_y+face_h),
        #               (0, 255, 0), 2)
        detect_eye = eye.detectMultiScale(img2, 1.2, 1)
        for (eye_x, eye_y, eye_z, eye_h) in detect_eye:
            if eye_y > 0.3 * face_h:
                continue
            if eye_h > 0.4 * face_h:
                continue

            # cv2.rectangle(frame, (face_x+eye_x, face_y+eye_y), (face_x+eye_x+eye_z, face_y+eye_y+eye_h),
            #               (0, 255, 0), 2)
            eye1 = gray[face_y+eye_y:face_y+eye_y +
                        eye_h, face_x+eye_x:face_x+eye_x+eye_z]

            eye1 = cv2.GaussianBlur(eye1, (3, 3), 2)
            # cv2.imshow('eyeBlured', eye1)
            ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('Binary', binary)

            width, height = binary.shape
            # Crop top 40%of the image
            binary = binary[int(0.3 * height):height, :]

            opening = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, Kernal)  # Opening Morphology
            dilate = cv2.morphologyEx(
                opening, cv2.MORPH_DILATE, Kernal, iterations=4)  # Dilate Morphology
            # cv2.imshow('Dilated', dilate)

            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE,  # Find contours
                                                   cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(eye1, contours, 0, (255, 0, 0), 3)
            # cv2.imshow('eyeBlured', eye1)
            # print(len(contours))

            if len(contours) != 0:
                cnt = contours[0]
                M1 = cv2.moments(cnt)

                Cx1 = int(M1['m10'] / M1['m00'])  # Find center of the contour
                Cy1 = int(M1['m01'] / M1['m00'])
                # Number of pixels we cropped from the image
                croppedImagePixelLength = int(0.3*height)
                center1 = (int(Cx1+face_x+eye_x), int(Cy1+face_y +
                                                      eye_y + croppedImagePixelLength))  # Center coordinates
                cv2.circle(frame, center1, 2, (0, 255, 0), 2)
                # print(center1)

    if not ret:  # If frame is not read then exit
        break
    if cv2.waitKey(1) == ord('s'):  # While loop exit condition
        break
    cv2.imshow('Frame Image', frame)  # Show original Image

cap.release()
cv2.destroyAllWindows()
