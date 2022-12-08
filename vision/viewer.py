# Python code for Multiple Color Detection
import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
import argparse
import time
from frame_processor import get_circles


# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",
# 	help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=64,
# 	help="max buffer size")
# args = vars(ap.parse_args())

# pts = deque(maxlen=args["buffer"])

# # if a video path was not supplied, grab the reference
# # to the webcam
# if not args.get("video", False):
# 	vs = VideoStream(src=0).start()
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
# # allow the camera or video file to warm up
# time.sleep(2.0)

# greenLower = (15, 40, 40)
# greenUpper = (115, 255, 255)

# # Start a while loop
# while(1):

#     # Reading the video from the
#     # webcam in image frames
#     input_frame = vs.read()
    
#     circles, mask = get_circles(input_frame)

#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         cv2.circle(input_frame, 
#             center=(circles[0, 0], circles[0, 1]), 
#             radius=circles[0, 2], 
#             color=(0, 255, 0), 
#             thickness=2)

#     # Display the resulting frame, quit with q
#     cv2.imshow('frame', input_frame)
#     cv2.imwrite("color_mask.jpg", mask)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# # if we are not using a video file, stop the camera video stream
# if not args.get("video", False):
# 	vs.stop()
# # otherwise, release the camera
# else:
# 	vs.release()


# Reading the video from the
# webcam in image frames
input_frame = cv2.imread("tennis_ball.jpg")

circles, mask = get_circles(input_frame)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    cv2.circle(input_frame, 
        center=(circles[0, 0], circles[0, 1]), 
        radius=circles[0, 2], 
        color=(0, 255, 0), 
        thickness=10)

# Display the resulting frame, quit with q
cv2.imshow('frame', mask)
cv2.imwrite("color_mask_with_fill_with_circles.jpg", input_frame)

cv2.destroyAllWindows()