import sys

import cv2
import imutils
import numpy as np
import scipy.spatial as sci

import main.functions as fc


def select_center(event, x, y, a, b):
    global px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        px, py = x, y


def min_area(event, x, y, a, b):

    global area, pxS, pyS

    if event == cv2.EVENT_LBUTTONDOWN:
        pxS, pyS = x, y
        print(pxS, pyS)
    elif event == cv2.EVENT_LBUTTONUP:
        base = pxS - x
        height = pyS - y
        area = np.abs(base) * np.linalg.norm(height)
        print(area)


img, distance, load = fc.menu_option(sys)

h, w = img.shape[:2]

# Set an grey form of the image
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get threshold
ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imgray = cv2.GaussianBlur(imgray, (7, 7), 0)

# Obtain the new camera matrix and undistorted the image both grey and colored
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(load["mtx"], load["dist"], (w, h), 1, (w, h))
undistortedImgray = cv2.undistort(thresh, load["mtx"], load["dist"], None, newCameraMtx)
undistortedImg = cv2.undistort(img, load["mtx"], load["dist"], None, newCameraMtx)

_, conts, _ = cv2.findContours(undistortedImgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
copy = undistortedImg.copy()

# Compute min area object
cv2.namedWindow('select min area dim', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('select min area dim', min_area)
cv2.drawContours(copy, conts, -1, (255, 255, 0), 3)
cv2.resizeWindow('select min area dim', 1366, 768)
cv2.imshow('select min area dim', copy)
while 1:
    k = cv2.waitKey(33)
    if k == 13:    # Enter key to exit
        break

cv2.destroyWindow('select min area dim')
print(area)

# Select object center in the image
cv2.namedWindow('select object center', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('select object center', select_center)
cv2.drawContours(copy, conts, -1, (255, 255, 0), 3)
cv2.resizeWindow('select object center', 1366, 768)
cv2.imshow('select object center', copy)
while 1:
    k = cv2.waitKey(33)
    if k == 13:    # Enter key to exit
        break

cv2.destroyWindow('select object center')

# Compute the pixel to the real dimension
pixelsPerMetricX = load["mtx"][0][0] / distance
pixelsPerMetricY = load["mtx"][1][1] / distance

minMidPx = None
minMidPy = None
eucDist = None
contours = None

# Start the loop to compute the nearest area to the selected point
if px is not None and py is not None:
    k = 1
    for c in conts:
        # Find only the area greater than 'area'
        if cv2.contourArea(c) > area:
            orig = undistortedImg.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            (tl, tr, br, bl) = box

            (midX, midY) = fc.midpoint(tl, br)

            if minMidPx is None and minMidPy is None:
                minMidPx = midX
                minMidPy = midY
                contours = c
                eucDist = sci.distance.euclidean((midX, midY), (px, py))

            elif eucDist > sci.distance.euclidean((midX, midY), (px, py)):
                print('nearest found ' + k.__str__())
                minMidPy = midY
                minMidPx = midX
                contours = c
                eucDist = sci.distance.euclidean((midX, midY), (px, py))

        k = k + 1

    fc.print_area(contours, undistortedImg, pixelsPerMetricX, pixelsPerMetricY)

else:
    for c in conts:
        # if the contour is not sufficiently large, ignore it
        if 10000 < cv2.contourArea(c) < 100000:
            # compute the rotated bounding box of the contour
            fc.print_area(c, undistortedImg, pixelsPerMetricX, pixelsPerMetricY)

cv2.destroyAllWindows()
