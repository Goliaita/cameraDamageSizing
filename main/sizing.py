import numpy as np
import cv2
import glob

load = np.load("./images/calib.npz")

# Load one of the test images
img = cv2.imread("./1m/image-2018-07-30_13-44-51.jpg")
h, w = img.shape[:2]

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(load["mtx"], load["dist"], (w, h), 1, (w, h))
undistortedImg = cv2.undistort(img, load["mtx"], load["dist"], None, newCameraMtx)

# Crop the undistorted image
# x, y, w, h = roi
# undistortedImg = undistortedImg[y:y + h, x:x + w]

# Display the final result
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1366, 768)
cv2.imshow('image', np.hstack((img, undistortedImg)))
cv2.waitKey(10000)

cv2.destroyAllWindows()

