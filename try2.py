import numpy as np
import cv2 as cv
import argparse
#ก้อนนี้สร้าง backgroundsubtraction
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='MOVI0388.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)


def on_change(self):
    pass


cv.namedWindow('Params')
cv.createTrackbar('Kernel', 'Params', 1, 21, on_change)

cv.setTrackbarPos('Kernel', 'Params', 5)

pause = False

currWidth = 0
currHeight = 0

crop = 0

while True:
    if not pause:
        ret, frame = capture.read()
        if frame is None:
            break

        frame = frame[:, crop:1280 - crop]
        fgMask = backSub.apply(frame)

    ret, thresh1 = cv.threshold(fgMask, 120, 255, cv.THRESH_BINARY)

    ksize = cv.getTrackbarPos('Kernel', 'Params')
    kernel = np.ones((ksize, ksize), np.uint8)
    closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    # Find Canny edges
    edged = cv.Canny(opening, 30, 200)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(edged,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.imshow('Canny Edges', edged)

    im_with_keypoints = np.copy(frame)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv.drawContours(im_with_keypoints, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv.contourArea)
        rotrect = cv.minAreaRect(c)
        box = cv.boxPoints(rotrect)
        box = np.int0(box)

        (x, y), (width, height), angle = rotrect
        w = min(width, height)
        h = max(width, height)

        if w / h > 0.6:
            currWidth = max(currWidth, w)
            currHeight = max(currHeight, h)
            print(currWidth, currHeight, currWidth * currHeight)

            # draw the biggest contour (c) in green
            cv.drawContours(im_with_keypoints, [box], 0, (0, 255, 0), 2)

    cv.rectangle(im_with_keypoints, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(im_with_keypoints, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Thres', opening)
    cv.imshow("Keypoints", im_with_keypoints)



    keyboard = cv.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('s'):
        pause = not pause