# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pdf2image import convert_from_path


class PanWindow(object):

    def __init__(self, img, windowName='PanZoomWindow', windowDim=[400, 600], onLeftClickFunction=None):
        self.WINDOW_NAME = windowName
        self.H_TRACKBAR_NAME = 'y'
        self.V_TRACKBAR_NAME = 'x'
        self.img = img
        self.windowDim = windowDim
        self.onLeftClickFunction = onLeftClickFunction
        # self.TRACKBAR_TICKS = 1000
        self.imH = img.shape[0]
        self.imW = img.shape[1]
        self.panState = PanState(img.shape, windowDim, self)
        self.lButtonDownLoc = None
        # cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
        cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME,
                           0, self.imH, self.onHTrackbarMove)
        cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME,
                           0, self.imW, self.onVTrackbarMove)

    def onMouse(self, event, x, y, *atr):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordsInDisplayedImage = np.array([y, x])
            if np.any(coordsInDisplayedImage < 0) or np.any(coordsInDisplayedImage > self.windowDim):
                print("you clicked outside the image area")
            else:
                print(
                    f"you clicked on {coordsInDisplayedImage} within the displayed image")
                coordsInFullImage = self.panState.ul + coordsInDisplayedImage

                print(
                    f"this is {coordsInFullImage} in the actual image")

                print(
                    f"this pixel holds {self.img[coordsInFullImage[0], coordsInFullImage[1]]}")

                if self.onLeftClickFunction is not None:
                    self.onLeftClickFunction(
                        coordsInFullImage[0], coordsInFullImage[1])

    def onVTrackbarMove(self, tickPosition):
        self.panState.setYFractionOffset(
            float(tickPosition)/self.imW)

    def onHTrackbarMove(self, tickPosition):
        self.panState.setXFractionOffset(
            float(tickPosition)/self.imH)

    def redrawImage(self):
        pzs = self.panState
        cv2.imshow(self.WINDOW_NAME,
                   self.img[pzs.ul[0]:pzs.ul[0]+self.windowDim[0], pzs.ul[1]:pzs.ul[1]+self.windowDim[1]])


class PanState(object):
    def __init__(self, imShape, windowDim, parentWindow):
        # upper left of the rectangle (expressed as y,x)
        self.ul = np.array([0, 0])

        self.windowDim = windowDim
        self.imShape = np.array(imShape[0:2])
        self.parentWindow = parentWindow

    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll outside the image. """
        # print("in self.ul: %s shape: %s"%(self.ul))
        self.ul = np.maximum(0, np.minimum(
            self.ul, (self.imShape-self.windowDim)))
        self.parentWindow.redrawImage()

    def setYFractionOffset(self, fraction):
        self.ul[0] = int(
            round((self.imShape[0] - self.windowDim[0]) * fraction))
        self._fixBoundsAndDraw()

    def setXFractionOffset(self, fraction):
        self.ul[1] = int(
            round((self.imShape[1] - self.windowDim[1]) * fraction))
        self._fixBoundsAndDraw()


if __name__ == "__main__":
    images = convert_from_path(
        "test_files/resume.pdf", poppler_path=f"C://Users//LENOVO//Downloads//poppler-21.11.0//Library//bin")

    images = list(
        map(
            lambda image: np.asarray(image),
            images
        )
    )

    # infile = "test_files/test_image.jpg"
    myImage = cv2.vconcat(images)
    # cv2.imshow("test", myImage)
    # cv2.imread(infile, cv2.IMREAD_ANYCOLOR)
    window = PanWindow(myImage, "test window", [600, myImage.shape[1]])
    key = -1
    while key != ord('q') and key != 27 and cv2.getWindowProperty(window.WINDOW_NAME, 0) >= 0:  # 27 = escape key
        # the OpenCV window won't display until you call cv2.waitKey()
        key = cv2.waitKey(5)  # User can press 'q' or ESC to exit.
    cv2.destroyAllWindows()
