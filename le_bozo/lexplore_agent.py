from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.ball_detector import get_circles
import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt

from numpy import unravel_index

class LeXploreAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        
        trans = self.vehicle.transform
        depth_img = self.front_depth_camera.data

        x_t = trans.location.x
        z_t = trans.location.z
        psi = trans.rotation.yaw

        # if self.time_counter % 250:
        #     return VehicleControl()

        if trans and depth_img is not None:
            depth_img = depth_img.copy()
            depth_img[0:len(depth_img)//2] = 0

            u_ball, v_ball = unravel_index(depth_img.argmax(), depth_img.shape)

            FOV = 69.39
            center_u = 72
            offset = v_ball - center_u
            theta = (FOV/center_u) * offset

            dist = depth_img[u_ball][v_ball]
            depth = abs(dist*np.cos(theta))

            x_ball = x_t + dist*np.sin(psi+theta)
            z_ball = z_t - dist*np.cos(psi+theta)

            error = np.array([depth/10, theta/180])
            kp = np.array([[0.15, 0], [0, 1]])

            t, s = kp@error

            print(f'theta: {theta}, dist: {dist}, u: {u_ball}, v: {v_ball}, x: {x_ball}, z: {z_ball}')
            print(f'throttle: {t}, steering: {s}\n')

            return VehicleControl(throttle=t, steering=s+0.25)

            # return VehicleControl(throttle = 0.01, steering=0.25)
        return VehicleControl(steering=0.25)