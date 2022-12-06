from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.ball_detector import get_circles
from ROAR.le_bozo.gripper_client import commandGripper
import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class FetchAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

        #CV and ball finding config
        self.prev_depth_img = None
        self.prev_finds = [0]*10 #running average of ball points
        self.circles = None
        self.ball_loc = None

        #tolerance parameter
        self.epsilon = 0.1

        #state parameter
        curr_state = "exploring"

        #x z coordinate of base
        self.base_trans = self.vehicle.transform

        #turning back around for resetting timer and gripping states
        self.gripper_clock = 10
        self.turning_clock = 10

        gripper_activated = True #toggle for activating gripper
        if gripper_activated:
            try:
                commandGripper("open")
                print("Gripper successfully opened.")
            except:
                print("Communication to gripper failed!")
                return VehicleControl(steering = 0.25)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        trans = self.vehicle.transform
        rgb_img = self.front_rgb_camera.data
        depth_img = self.front_depth_camera.data
        velocity = self.vehicle.velocity

        #prelaunch checks
        if trans is None:
            print("Vehicle localization lost!")
            return VehicleControl(steering = 0.25)
        if rgb_img is None:
            print("Color image lost!")
            return VehicleControl(steering = 0.25)
        if depth_img is None:
            print("Vehicle LIDAR Matrix lost!")
            return VehicleControl(steering = 0.25)
        if velocity is None:
            print("Vehicle velocity lost!")
            return VehicleControl(steering = 0.25)

        #vehicle variables
        x_t = trans.location.x #x location of the bot
        z_t = trans.location.z #z location of the bot
        psi = trans.rotation.yaw #yaw of the bot relative to the spatial frame

        #vision variables
        FOV = 69.39 #hardcoded field of view for the camera
        center_u = 72 #hardcoded center of screen (columnwise)
        
            
        if self.curr_state == "exploring":
            
            #seeing if we found a ball or not, if so, change controller
            circles, mask = get_circles(rgb_img)
            if circles is not None:
                print("Ball found, changing to approach controller")
                self.ball_loc, theta, depth = get_ball_loc(circles, depth_img, x_t, z_t, psi)
                print("theta img:", theta)
                self.curr_state = "ball"
                return VehicleControl(steering = 0.25)

            print("Exploring...")

            #run Explore Agent

            #downsampling depth image
            depth_img = depth_img.copy()
            depth_img[0:int(len(depth_img)//2)] = 0

            #updating based on older depth image
            if self.old is not None:
                depth_img += .25 * self.old
            
            #cursed convolution to compress depth image and find farthest depth
            k = np.ones((40, 20))
            depth_img_convolved = ndimage.convolve(depth_img, k, mode='nearest', cval=0.0)
            u_ball, v_ball = np.unravel_index(depth_img_convolved.argmax(), depth_img.shape)

            #compute the projected pixel offset and angular error
            offset = v_ball - center_u
            theta = (FOV/center_u) * offset

            #compute the distance and depth of the farthest LIDAR point
            dist = depth_img[u_ball][v_ball]
            depth = abs(dist*np.cos(theta))

            #believed target coordinates
            x_ball = x_t + dist*np.sin(psi+theta)
            z_ball = z_t - dist*np.cos(psi+theta)

            #PI controller
            target_vel = 0.1
            v_scalar = (velocity.x*velocity.x + velocity.z*velocity.z) ** (1/2)
            error = np.array([depth, theta/180, v_scalar - target_vel])
            kp = np.array([[0.015, 0, -2], [0, 2, 0], [0, 0, 0]])
            t, s, v = kp@error

            print(f'theta: {theta}, dist: {dist}, u: {u_ball}, v: {v_ball}, x: {x_ball}, z: {z_ball}')
            print(f'throttle: {t}, steering: {s}, velocity: {v_scalar}\n')

            self.old = depth_img #convolution stuff?

            return VehicleControl(throttle=t, steering=s+0.25)

        if self.curr_state == "ball":
            print("Approaching ball...")
            #run P Agent

            #TODO: Write this tomorrow!!!

        if self.curr_state == "grip":
            print("Gripping ball...")
            #close the gripper, wait for a bit

            #make sure the vehicle is at a complete stop
            if self.gripper_clock >= 0:
                self.gripper_clock -= 1
                return VehicleControl(steering = 0.25)
            else:
                self.gripper_clock = 10 #reset the clock for later
                commandGripper("close") #close the gripper
                self.curr_state = "return" #tell bot to return
                return VehicleControl(steering = 0.25) 

        if self.curr_state == "return":

            #run RTB Agent
            x_b, z_b = self.base_trans

            v_cs = np.array([-x_t, -z_t])
            v_cb = np.array([x_b - x_t, z_b - z_t])
            
            #TODO: Write episilon tolerance for theta and decrease steering override!
            if dist > self.epsilon:
                # travel to base
                print('Returning to base..')

                #finding the deflection between psi and theta (optimal vector to go home)
                theta_prime = angle_between(v_cb, v_cs)
                theta_prime_err = angle_between(np.array([0, -1]), v_cs)

                #deflection stored as theta_err
                theta = theta_prime + psi - theta_prime_err
                print("3d theta: ", theta)
                print(x_b, z_b)

                target_vel = 0.1
                v_scalar = (velocity.x*velocity.x + velocity.z*velocity.z) ** (1/2)

                error = np.array([dist, theta, v_scalar - target_vel])

                kp = np.array([[0.015, 0, -1], [0, 1, 0], [0, 0, 0]])

                t, s, v = kp@error
                return VehicleControl(throttle=t, steering=s+0.25)
        
            else:
                print("At base")
                self.curr_state = "dropoff"
                return VehicleControl(steering = 0.25)

        if self.curr_state == "dropoff":
            print("Releasing ball...")
            #open the gripper and rapidly reverse for a second

            #make sure the vehicle is at a complete stop
            if self.gripper_clock >= 0:
                self.gripper_clock -= 1
                return VehicleControl(steering = 0.25)
            else:
                self.gripper_clock = 10 #reset the clock for later
                commandGripper("open") #open the gripper
                self.curr_state = "reset" #tell bot to reset
                return VehicleControl(throttle = -1, steering = 0.25) #rapid straight reverse

        if self.curr_state == "reset":
            print("Resetting position...")
            #turn back around for a few seconds
            if self.turning_clock >= 0:
                self.turning_clock -=1
                return VehicleControl(steering = 0.45) #strong right turn
            else:
                self.turning_clock = 10
                self.curr_state = "exploring" #return back to exploration phase
                return VehicleControl(steering = 0.25) #straighten out

            
