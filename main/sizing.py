import sys

import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# TODO implement data parameter from cmd

usage = 'usage: [-d distance](if not inserted is given as 1 meter)' \
               '[-px x position pixels](find object in that area)' \
               '[-py y position pixels](find object in that area)'

if len(sys.argv) % 2 == 0:
    exit(usage)
else:
    for i, arg in enumerate(sys.argv):
        if arg == '-d':
            try:
                distance = float(sys.argv[i + 1])
            except ValueError:
                exit(usage)

        elif arg == '-px':
            try:
                positionX = float(sys.argv[i + 1])
            except ValueError:
                exit(usage)

        elif arg == '-py':
            try:
                positionY = float(sys.argv[i + 1])
            except ValueError:
                exit(usage)

try:
    distance
except NameError:
    distance = 1

# Load calibration data
load = np.load("./calib4.npz")

# Load one of the test images
img = cv2.imread("./4m/test1.jpg")
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

# Get Contours
_, conts, _ = cv2.findContours(undistortedImgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

copy = undistortedImg.copy()

# Show basic photo
cv2.drawContours(copy, conts, -1, (255, 255, 0), 3)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1366, 768)
cv2.imshow('image', copy)
cv2.waitKey(20000)

pixelsPerMetricX = load["mtx"][0][0] / distance
pixelsPerMetricY = load["mtx"][1][1] / distance

for c in conts:

    # if the contour is not sufficiently large, ignore it
    if 100 < cv2.contourArea(c) < 150:

        # compute the rotated bounding box of the contour
        orig = undistortedImg.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        # box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        # for (x, y) in box:
        #     cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        print(dA)
        print(dB)

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        dimA = dA / pixelsPerMetricX * 1000
        dimB = dB / pixelsPerMetricY * 1000

        # draw the object sizes on the image
        cv2.putText(orig, "{:.2f}mm".format(dimB),
                    (int(tltrX - 150), int(tltrY)), cv2.QT_FONT_NORMAL,
                    1.5, (255, 255, 255), 2)
        cv2.putText(orig, "{:.2f}mm".format(dimA),
                    (int(trbrX), int(trbrY - 30)), cv2.QT_FONT_NORMAL,
                    1.5, (255, 255, 255), 2)

        # Display the final result
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1366, 768)
        cv2.imshow('image', orig)
        cv2.waitKey(20000)

cv2.destroyAllWindows()
