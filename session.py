import cv2
import numpy as np
import argparse

# back_sub = cv2.bgsegm.createBackgroundSubtractorMOG()
# # back_sub = cv2.createBackgroundSubtractorMOG2()
# # back_sub = cv2.bgsegm.createBackgroundSubtractorGMG()
#
# capture = cv2.VideoCapture("test_files/video.mp4")
#
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
#
#     fg_mask = back_sub.apply(frame)
#     cv2.imshow("Frame", frame)
#     cv2.imshow("FG MASK", fg_mask)
#     key = cv2.waitKey(30)
#     if key == "q" or key == 27:
#         break

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

ret, frame1 = cap.read()
ret, frame2 = cap.read()


while True:
    ret, frame = cap.read()
    d = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    cv2.imshow("th", th)

    # dilate = cv2.dilate(th, np.ones((7, 7), np.uint8), iterations=1)
    dilate = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=5)
    # dilate = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    cv2.imshow("d", dilate)

    c, h = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, c, -1, (0, 255, 0), 2)

    cv2.imshow("inter", frame1)

    key = cv2.waitKey(30)

    if key == "q" or key == 27:
        break

    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()




























