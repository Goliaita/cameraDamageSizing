import sys

import cv2
import imutils
import numpy as np
import scipy.spatial as sci


def printarea(_c):
    print(_c)
    orig = undistortedImg.copy()
    box = cv2.minAreaRect(_c)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    # box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

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

    dA = sci.distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = sci.distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

    print(midpoint(tr, bl))
    print(midpoint(tl, br))
    print(sci.distance.euclidean(midpoint(tr, bl), (1359, 2355)))

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
    cv2.resizeWindow('image', 768, 1366)
    cv2.imshow('image', orig)
    cv2.waitKey(20000)


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# TODO implement data parameter from cmd

usage = 'usage: [-d distance](if not inserted is given as 1 meter)' \
               '[-px x position pixels](find object in that area)' \
               '[-py y position pixels](find object in that area)'


px = None
py = None

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
                px = float(sys.argv[i + 1])
            except ValueError:
                exit(usage)

        elif arg == '-py':
            try:
                py = float(sys.argv[i + 1])
            except ValueError:
                exit(usage)

try:
    distance
except NameError:
    distance = 1

# Load calibration data
load = np.load("./nexus5x1mP.npz")

# Load one of the test images
img = cv2.imread("./nexus 5x 1m portrait/test1.jpg")
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
cv2.resizeWindow('image', 768, 1366)
cv2.imshow('image', copy)
cv2.waitKey(20000)


pixelsPerMetricX = load["mtx"][0][0] / distance
pixelsPerMetricY = load["mtx"][1][1] / distance

minMidPx = None
minMidPy = None
eucDist = None
contours = None

if px is not None and py is not None:
    k = 1
    for c in conts:
        if cv2.contourArea(c) > 10:
            orig = undistortedImg.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            (tl, tr, br, bl) = box

            (midX, midY) = midpoint(tl, br)

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

else:
    for c in conts:

        # if the contour is not sufficiently large, ignore it
        if 500 < cv2.contourArea(c) < 100000:

            # compute the rotated bounding box of the contour
            printarea(c)

printarea(contours)
cv2.destroyAllWindows()
