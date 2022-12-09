from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.car_utils import *
import time

import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from numpy import unravel_index
import skimage.measure

from ROAR.le_bozo.gripper_client import *

class TowMaterAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.prev_depth_img = None
        self.prev_finds = np.zeros(6)
        self.circles = None
        self.ball_loc = None
        self.depth = None
        self.epsilon = 0.3
        self.gripper_activated = True
        if self.gripper_activated:
            print("opening gripper")
            commandGripper("open")
        self.home_trans = self.vehicle.transform
        self.rtb = False
        self.turn_timer = 100
        self.turn_psi = 0

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


            if self.rtb:
                print("GOING TO TURN")
                print("psi: " + str(psi))
                print("turn psi: " + str(self.turn_psi))

                theta, dist = get_reconstruction_td([x_t, z_t], [self.home_trans.location.x, self.home_trans.location.z], psi)

                if self.gripper_activated:
                        commandGripper("close")

                if abs((psi%360)-(theta%360)) < 20:
                    return VehicleControl(throttle = 0.05, steering = 0.9)
                else:
                    print("car x, y: ", x_t, z_t)
                    print("Distance to home is: " + str(dist))

                    if abs(dist) > self.epsilon:
                        # travel to ball
                        # print(f'Default to x: {x_t}, z: {z_t} ({dist}m away)')
                        print("Loc theta is: " + str(theta))

                        # print(f'x_b: {x_b}, z_b: {z_b}, theta: {theta}')

                        target_vel = 0.07
                        v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                        error = np.array([dist, theta/180, (v_scalar - target_vel)])

                        kp = np.array([[0.015, 0, -1], [0, 5, 0], [0, 0, 0]])

                        t, s, v = kp@error
                        return VehicleControl(throttle=t, steering=s+0.25)
                        
                    else:
                        # release
                        print('at home')
                        if self.gripper_activated:
                            commandGripper("open")
                        return VehicleControl(steering = 0.25)


            
            circles, mask = get_circles(rgb_img)

            if circles is not None:
                # update ball_loc
                print('Live CV')
                self.ball_loc, self.depth, theta = get_ball_loc(circles, depth_img, x_t, z_t, psi)
                print("CV theta is: " + str(theta))
                print("Live CV depth: ", self.depth)

                target_vel = 0.1
                v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)
                error = np.array([self.depth, theta/180, v_scalar - target_vel])
                kp = np.array([[0.015, 0, -1.5], [0, 5, 0], [0, 0, 0]])
                t, s, v = kp@error
                return VehicleControl(throttle=t, steering=s+0.25)
            
            elif self.ball_loc is not None:

                theta, dist = get_reconstruction_td([x_t, z_t], self.ball_loc, psi)

                print("car x, y: ", x_t, z_t)
                print("Distance to ball is: " + str(dist))

                if abs(dist) > self.epsilon:
                    # travel to ball
                    print(f'Last Location {self.ball_loc}')
                    # print(f'Default to x: {x_t}, z: {z_t} ({dist}m away)')
                    print("Loc theta is: " + str(theta))

                    # print(f'x_b: {x_b}, z_b: {z_b}, theta: {theta}')

                    target_vel = 0.07
                    v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                    error = np.array([dist, theta/180, (v_scalar - target_vel)])

                    kp = np.array([[0.015, 0, -1], [0, 5, 0], [0, 0, 0]])

                    t, s, v = kp@error
                    return VehicleControl(throttle=t, steering=s+0.25)
                    
                else:
                    # grip and go home
                    print('at ball')
                    self.turn_psi = 180+ psi
                    if self.gripper_activated:
                        commandGripper("close")
                    time.sleep(1)
                    self.rtb = True
                    return VehicleControl(throttle = 0.2, steering = 0.5)

            else:
                # look for ball
                print('exploring')


                depth_img = depth_img.copy()
                depth_img[0:int(len(depth_img)//2)] = 0

                if self.prev_depth_img is not None:
                    depth_img += 0 * self.prev_depth_img
                
                k = np.ones((40, 20))
                depth_img_pooled = skimage.measure.block_reduce(depth_img, (16,16), np.min)
                depth_img_pooled = np.kron(depth_img_pooled, np.ones((16,16)))
                u_ball, v_ball = unravel_index(depth_img_pooled.argmax(), depth_img_pooled.shape)

                FOV = 69.39
                center_u = 72
                offset = v_ball - center_u
                theta = (FOV/(2*center_u)) * offset

                theta_rad = theta * np.pi/180.0
                psi_rad = psi * np.pi/180.0

                dist = depth_img[u_ball][v_ball]
                depth = abs(dist*np.cos(theta_rad))

                target_vel = 0.1
                v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                error = np.array([depth, theta/180, v_scalar - target_vel])
                kp = np.array([[0.015, 0, -1], [0, 7, 0], [0, 0, 0]])

                t, s, v = kp@error

                self.prev_depth_img = depth_img

                return VehicleControl(throttle=t, steering=s+0.25)
            cv2.imshow("img", mask)

        return VehicleControl(steering=0.25)