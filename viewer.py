# Python code for Multiple Color Detection
import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# Start a while loop
while(1):
    # Reading the video from the
    # webcam in image frames
    input_frame = vs.read()
    frame = input_frame.copy()

    blurred = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.GaussianBlur(mask, (11, 11), 2, 2)


    # find contours in the mask and initialize the current
	# (x, y) center of the ball
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    circles = cv2.HoughCircles(mask, 
        cv2.HOUGH_GRADIENT, 
        1, 
        mask.shape[0] / 8, 
        param1=100, 
        param2=20, 
        minRadius=0, 
        maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cv2.circle(frame, 
            center=(circles[0, 0], circles[0, 1]), 
            radius=circles[0, 2], 
            color=(0, 255, 0), 
            thickness=2)


    # Display the resulting frame, quit with q
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()

cv2.destroyAllWindows()