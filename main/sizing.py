import argparse

import cv2
import imutils
import numpy as np
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# Load calibration data
load = np.load("./calib3.npz")
print(load["mtx"][0][0])
a = load["mtx"][0][0] * 4.6 / 3296

# Load one of the test images
img = cv2.imread("./3m/image-2018-07-30_13-51-33.jpg")
h, w = img.shape[:2]

blurred = cv2.pyrMeanShiftFiltering(img, 31, 91)

bound = 30
inc = np.array([46.7, 36.1, 30.6])
lower = inc - bound
upper = inc + bound

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imgray = cv2.GaussianBlur(imgray, (7, 7), 0)

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(load["mtx"], load["dist"], (w, h), 1, (w, h))
undistortedImgray = cv2.undistort(thresh, load["mtx"], load["dist"], None, newCameraMtx)
undistortedImg = cv2.undistort(img, load["mtx"], load["dist"], None, newCameraMtx)

_, conts, _ = cv2.findContours(undistortedImgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

copy = undistortedImg.copy()

cv2.drawContours(copy, conts, -1, (255, 255, 0), 3)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1366, 768)
cv2.imshow('image', copy)
cv2.waitKey(20000)

pixelsPerMetric = a

for c in conts:

    # if the contour is not sufficiently large, ignore it
    if 10 < cv2.contourArea(c) < 1000:
        print(c)

        # compute the rotated bounding box of the contour
        orig = undistortedImg.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

# hull = cv2.convexHull(cnt, returnPoints=ret)


# x, y, w, h = cv2.boundingRect(ret)
# cv2.rectangle(undistortedImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

# for i in range(0, len(contours)):
# cv2.drawContours(undistortedImg, contours, -1, (255, 255, 0), 3)

# Display the final result
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1366, 768)
        cv2.imshow('image', orig)
        cv2.waitKey(20000)

cv2.destroyAllWindows()
