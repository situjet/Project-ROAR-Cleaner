# Python code for Multiple Color Detection
import numpy as np
import cv2

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

def get_circles(input_frame):
    frame = input_frame.copy()

    blurred = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.GaussianBlur(mask, (11, 11), 2, 2)

    circles = cv2.HoughCircles(mask, 
        cv2.HOUGH_GRADIENT, 
        1, 
        mask.shape[0] / 8, 
        param1=100, 
        param2=20, 
        minRadius=0, 
        maxRadius=0)

    return circles

