import numpy as np
import cv2
import glob
# prova
# Define the chess board rows and columns
rows = 4
cols = 5
# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.005)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

print("loading images")

# set different distance
meter1 = "./1m/*.jpg"
meter2 = "./2m/*.jpg"
meter3 = "./3m/*.jpg"
meter4 = "./4m/*.jpg"
cameraD = "./camera davide/*.jpg"
cameraD4m = "./nexus 5x 4m/*.jpg"
nexus1m = "./nexus 5x 1m portrait/*.jpg"
nexus5xportrait1mfront = "./nexus5xportrait1mfront/*.jpg"
honor1m = "./honor1m/*.jpg"
honor2m = "./honor2m/*.jpg"
honor5m = "./honor5m/*.jpg"


k = 1

for path in glob.glob(honor2m):

    print("image " + k.__str__() + " loaded")

    k = k + 1

    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret:
        print("true")

        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

    # Display the image

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 768, 1366)
    # cv2.imshow('image', img)
    # cv2.waitKey(10000)

# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
print(dist)
np.savez("./honor2m" + ".npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))

cv2.destroyAllWindows()
