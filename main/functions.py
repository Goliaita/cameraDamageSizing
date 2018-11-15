import cv2
import imutils
import numpy as np
import scipy.spatial as sci


def print_area(_c, _orig, pixelsPerMetricX, pixelsPerMetricY):
    _box = cv2.minAreaRect(_c)
    _box = cv2.boxPoints(_box) if imutils.is_cv2() else cv2.boxPoints(_box)
    _box = np.array(_box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    # box = perspective.order_points(box)
    cv2.drawContours(_orig, [_box.astype("int")], -1, (0, 255, 0), 2)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (_tl, _tr, _br, _bl) = _box
    (tltrX, tltrY) = midpoint(_tl, _tr)
    (blbrX, blbrY) = midpoint(_bl, _br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(_tl, _bl)
    (trbrX, trbrY) = midpoint(_tr, _br)

    _dA = sci.distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
    _dB = sci.distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

    _centerX = (trbrX + tlblX) / 2
    _centerY = (trbrY + tlblY) / 2

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    print(_dA * _dB)
    _dimA = _dA / pixelsPerMetricX * 1000
    _dimB = _dB / pixelsPerMetricY * 1000

    # draw the object sizes on the image
    cv2.putText(_orig, "{:.2f}mm".format(_dimB),
                (int(tltrX - 150), int(tltrY)), cv2.QT_FONT_NORMAL,
                1.5, (255, 255, 255), 2)
    cv2.putText(_orig, "{:.2f}pixel X - {:.2f}pixel Y".format(_centerX, _centerY),
                (int(tltrX + 150), int(tltrY)), cv2.QT_FONT_NORMAL,
                1.5, (255, 255, 255), 2)
    cv2.putText(_orig, "{:.2f}mm".format(_dimA),
                (int(trbrX), int(trbrY - 30)), cv2.QT_FONT_NORMAL,
                1.5, (255, 255, 255), 2)

    # Display the final result
    cv2.namedWindow('area', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('area', 1366, 768)
    cv2.imshow('area', _orig)
    _k = None
    while 1:
        _k = cv2.waitKey(33)
        if _k == 13:  # Enter key to exit
            break


""""""


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


""""""


def menu_option(sys):

    usage = 'usage: [-d distance]\n'  '[-a area dimension]\n' \
            '[-f \'path to npz file\']\n' '[-p \'path to photo\']\n'

    photo = None
    file_npz = None
    d = None
    a = None

    if len(sys.argv) % 2 == 0:
        exit(usage)
    else:
        for i, arg in enumerate(sys.argv):
            if arg == '-d':
                d = sys.argv[i + 1]

            if arg == '-a':
                a = sys.argv[i + 1]

            elif arg == '-p':
                try:
                    photo = sys.argv[i + 1]
                except ValueError:
                    exit(usage)

            elif arg == '-f':
                try:
                    file_npz = sys.argv[i + 1]
                except ValueError:
                    exit(usage)

    try:
        if d is not None:
            distance = float(d)
        else:
            distance = 1
    except NameError:
        # Distance setted to 1
        distance = 1

    # Load calibration data<
    try:
        load = np.load(file_npz)
    except AttributeError:
        load = np.load("./honor2m.npz")

    # Load one of the test images
    img = cv2.imread(photo)
    if img is None:
        img = cv2.imread("./honor2m/test2.jpg")

    return img, distance, load
