from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.car_utils import *
import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from numpy import unravel_index

class LeXploreAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.old = None

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        
        trans = self.vehicle.transform
        depth_img = self.front_depth_camera.data

        x_t = trans.location.x
        z_t = trans.location.z
        psi = trans.rotation.yaw
        velocity = self.vehicle.velocity

        # if self.time_counter % 250:
        #     return VehicleControl()
        print(f'velocity: {velocity}')

        if trans and depth_img is not None:
            depth_img = depth_img.copy()
            # depth_img[:,0:int(.2*len(depth_img[0]))] = 0
            # depth_img[:,int(.8*len(depth_img[0])):] =0
            depth_img[0:int(len(depth_img)//2)] = 0

            # plt.imshow(depth_img)
            # plt.show()

            if self.old is not None:
                depth_img += .25 * self.old

            k = np.ones((40, 20))
            depth_img_convolved = ndimage.convolve(depth_img, k, mode='nearest', cval=0.0)
            u_ball, v_ball = unravel_index(depth_img_convolved.argmax(), depth_img.shape)

            # u_ball, v_ball = unravel_index(depth_img.argmax(), depth_img.shape)

            FOV = 69.39
            center_u = 72
            offset = v_ball - center_u
            theta = (FOV/center_u) * offset

            dist = depth_img[u_ball][v_ball]
            depth = abs(dist*np.cos(theta))

            x_ball = x_t + dist*np.sin(psi+theta)
            z_ball = z_t - dist*np.cos(psi+theta)

            target_vel = 0.1
            v_scalar = (velocity.x*velocity.x + velocity.z*velocity.z) ** (1/2)

            error = np.array([depth, theta/180, v_scalar - target_vel])
            kp = np.array([[0.015, 0, -2], [0, 2, 0], [0, 0, 0]])
            # kp = np.array([[-0.15, 0, 1.5], [0, -7, 0], [0, 0, 0]])

            t, s, v = kp@error

            print(f'theta: {theta}, dist: {dist}, u: {u_ball}, v: {v_ball}, x: {x_ball}, z: {z_ball}')
            print(f'throttle: {t}, steering: {s}, velocity: {v_scalar}\n')

            # plt.imshow(depth_img)
            # plt.show()
            self.old = depth_img

            return VehicleControl(throttle=t, steering=s+0.25)

            # return VehicleControl(throttle = 0.01, steering=0.25)
        return VehicleControl(steering=0.25)