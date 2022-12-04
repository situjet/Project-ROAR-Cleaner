# Python code for Multiple Color Detection
import numpy as np
import cv2

# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
greenLower = (0.09*256, 0.60*256, 0.20*256)
greenUpper = (0.14*256, 1.00*256, 1.00*256)

def get_circles(input_frame):
    
    frame = input_frame.copy()

    blurred = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    open_struct = cv2.getStructuringElement(cv2.MORPH_RECT,(19,19))
    close_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_struct)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_struct)
    
    mask = cv2.GaussianBlur(mask, (15, 15), 2, 2)

    circles = cv2.HoughCircles(mask, 
        cv2.HOUGH_GRADIENT, 
        1, 
        mask.shape[0] / 4, 
        param1=20, 
        param2=20, 
        minRadius=0, 
        maxRadius=0)

    return circles, mask
    