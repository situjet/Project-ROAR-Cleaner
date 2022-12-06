from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.ball_detector import *
import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from numpy import unravel_index

class TowMaterAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.prev_depth_img = None
        self.prev_finds = np.zeros(6)
        self.circles = None
        self.ball_loc = None
        self.epsilon = 0.1

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        
        trans = self.vehicle.transform
        rgb_img = self.front_rgb_camera.data
        depth_img = self.front_depth_camera.data
        v_t = self.vehicle.velocity

        if trans is not None and rgb_img is not None and depth_img is not None and v_t is not None:
            x_t = trans.location.x
            z_t = trans.location.z
            psi = trans.rotation.yaw
            
            circles, mask = get_circles(rgb_img)
            if circles is not None:
                # update ball_loc
                print('ball')
                self.ball_loc, theta, depth = get_ball_loc(circles, depth_img, x_t, z_t, psi)
                print("theta img:", theta)
            
            if self.ball_loc is not None:
                x_b, z_b = self.ball_loc

                v_cs = np.array([-x_t, -z_t])
                v_cb = np.array([x_b - x_t, z_b - z_t])
                dist = np.linalg.norm(v_cb)

                if dist > self.epsilon:
                    # travel to ball
                    print('travelling to ball')

                    #finding the deflection between psi and theta (optimal vector to go home)
                    theta_prime = angle_between(v_cb, v_cs)
                    theta_prime_err = angle_between(np.array([0, -1]), v_cs)

                    #deflection stored as theta_err
                    theta = theta_prime + psi - theta_prime_err
                    print("3d theta: ", theta)
                    print(x_b, z_b)

                    target_vel = 0.1
                    v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                    error = np.array([dist, theta, v_scalar - target_vel])

                    kp = np.array([[0.015, 0, -1], [0, 1, 0], [0, 0, 0]])

                    t, s, v = kp@error
                    # return VehicleControl(throttle=t, steering=s+0.25)
                else:
                    # grip and go home
                    print('at ball')
            else:
                # look for ball
                print('exploring')

                depth_img = depth_img.copy()
                depth_img[0:int(len(depth_img)//1.5)] = 0

                if self.prev_depth_img is not None:
                    depth_img += 0 * self.prev_depth_img
                
                k = np.ones((40, 20))
                depth_img_convolved = ndimage.convolve(depth_img, k, mode='nearest', cval=0.0)
                u_ball, v_ball = unravel_index(depth_img_convolved.argmax(), depth_img.shape)

                FOV = 69.39
                center_u = 72
                offset = v_ball - center_u
                theta = (FOV/(1*center_u)) * offset

                dist = depth_img[u_ball][v_ball]
                depth = abs(dist*np.cos(theta))

                x_ball = x_t + dist*np.sin(psi+theta)
                z_ball = z_t - dist*np.cos(psi+theta)

                target_vel = 0.1
                v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                error = np.array([depth, theta/180, v_scalar - target_vel])
                kp = np.array([[0.15, 0, -1], [0, 5, 0], [0, 0, 0]])

                t, s, v = kp@error

                self.prev_depth_img = depth_img

                # return VehicleControl(throttle=t, steering=s+0.25)

        return VehicleControl(steering=0.25)