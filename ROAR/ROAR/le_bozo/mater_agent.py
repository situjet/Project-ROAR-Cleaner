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

class MaterAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.prev_depth_img = None
        self.prev_finds = np.zeros(6)
        self.circles = None

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
                self.prev_finds = np.append(self.prev_finds, 1)
                self.circles = circles
            else:
                self.prev_finds = np.append(self.prev_finds, 0)
            
            self.prev_finds = self.prev_finds[1:]

            if 1 in self.prev_finds:
                print('ball')
                # go towards ball
                circles2 = np.round(self.circles[0, :]).astype("int")
                cv2.circle(mask, center=(circles2[0, 0], circles2[0, 1]), radius=circles2[0, 2], color=(0, 255, 0), thickness=2)
                u_ball, v_ball = int(self.circles[0][0][1]//5), int(self.circles[0][0][0]//5)
                FOV = 69.39
                center_u = 72
                offset = v_ball - center_u
                theta = (FOV/(2*center_u)) * offset

                dist = depth_img[u_ball][v_ball]
                depth = abs(dist*np.cos(theta))
                offset_h = dist*np.sin(theta)

                x_ball = x_t + dist*np.sin(psi+theta)
                z_ball = z_t - dist*np.cos(psi+theta)

                target_vel = 0.1
                v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                error = np.array([depth, theta/180, v_scalar - target_vel])

                kp = np.array([[0.015, 0, -1], [0, 3, 0], [0, 0, 0]])

                t, s, v = kp@error
                return VehicleControl(throttle=t, steering=s+0.25)
            else:
                # look for ball
                print('no ball')
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
                theta = (FOV/(2*center_u)) * offset

                dist = depth_img[u_ball][v_ball]
                depth = abs(dist*np.cos(theta))

                x_ball = x_t + dist*np.sin(psi+theta)
                z_ball = z_t - dist*np.cos(psi+theta)

                target_vel = 0.1
                v_scalar = (v_t.x*v_t.x + v_t.z*v_t.z) ** (1/2)

                error = np.array([depth, theta/180, v_scalar - target_vel])
                kp = np.array([[0.015, 0, -1], [0, 3, 0], [0, 0, 0]])

                t, s, v = kp@error

                self.prev_depth_img = depth_img

                return VehicleControl(throttle=t, steering=s+0.25)

        return VehicleControl(steering=0.25)