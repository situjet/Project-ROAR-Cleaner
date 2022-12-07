# # Python code for Multiple Color Detection
# import numpy as np
# import cv2

# greenLower = (29, 86, 0)
# greenUpper = (64, 255, 255)

# def get_circles(input_frame):
#     frame = input_frame.copy()

#     blurred = cv2.medianBlur(frame, 5)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
#     mask = cv2.inRange(hsv, greenLower, greenUpper)
#     mask = cv2.GaussianBlur(mask, (11, 11), 2, 2)

#     circles = cv2.HoughCircles(mask, 
#         cv2.HOUGH_GRADIENT, 
#         1, 
#         mask.shape[0] / 8, 
#         param1=90, 
#         param2=27, 
#         minRadius=0, 
#         maxRadius=0)

#     return circles, mask


# Python code for Multiple Color Detection
import numpy as np
import cv2

# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
# greenLower = (0.09*256, 0.60*256, 0.20*256)
# greenUpper = (0.14*256, 1.00*256, 1.00*256)

# def get_circles(input_frame):
    
#     frame = input_frame.copy()

#     # frame[0:int(len(frame)//3)] = 0

#     blurred = cv2.medianBlur(frame, 5)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
#     mask = cv2.inRange(hsv, greenLower, greenUpper)

#     open_struct = cv2.getStructuringElement(cv2.MORPH_RECT,(19,19))
#     close_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_struct)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_struct)
    
#     mask = cv2.GaussianBlur(mask, (15, 15), 2, 2)

#     circles = cv2.HoughCircles(mask, 
#         cv2.HOUGH_GRADIENT, 
#         1, 
#         mask.shape[0] / 4, 
#         param1=20, 
#         param2=20, 
#         minRadius=0, 
#         maxRadius=0)

#     return circles, mask

greenLower = (0.09*256, 0.60*256, 0.20*256)
greenUpper = (0.15*256, 1.00*256, 1.00*256)

def get_circles(input_frame):
    
    frame = input_frame.copy()

    # frame[0:int(len(frame)//3)] = 0

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
        param1=40, 
        param2=10, 
        minRadius=0, 
        maxRadius=0)

    return circles, mask

def get_ball_loc(circles, depth_img, x_t, z_t, psi):
    circles2 = np.round(circles[0, :]).astype("int")
    # cv2.circle(mask, center=(circles2[0, 0], circles2[0, 1]), radius=circles2[0, 2], color=(0, 255, 0), thickness=2)
    u_ball, v_ball = int(circles[0][0][1]//5), int(circles[0][0][0]//5)
    FOV = 69.39
    center_u = 72
    offset = v_ball - center_u
    theta = (FOV/(2*center_u)) * offset

    theta_rad = theta * np.pi/180.0
    psi_rad = psi * np.pi/180.0

    dist = depth_img[u_ball][v_ball]
    depth = abs(dist*np.cos(theta_rad))

    x_b = x_t + dist*np.sin(psi_rad+theta_rad)
    z_b = z_t - dist*np.cos(psi_rad+theta_rad)

    return (x_b, z_b), depth, theta

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return rad*180/np.pi



