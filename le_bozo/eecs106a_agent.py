from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
from collections import deque
import numpy as np
from matplotlib import pyplot as plt

class EECS106Agent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        trans = self.vehicle.transform
        vel = self.vehicle.velocity
        control = self.vehicle.control
        # print(trans)
        if trans and vel and control:
            # print(trans, vel, control)
            x_t, z_t = trans.location.x, trans.location.z
            throttle_t, steering_t = control.get_steering(), control.get_throttle()

            target_x, target_z = 0, 1
            t = 0.9*(target_z - z_t)
            s = 0.1*(x_t - target_x)
            t = 0.01
            s = 0
            print(trans)
            if self.front_depth_camera.data is not None:
                print(self.front_depth_camera.data)
                plt.imshow(self.front_depth_camera.data)
                plt.show()
            # print(trans, t, s)
            # return VehicleControl(throttle = t, steering = s)
            
            
            # if curr_y < target_y:
            #     return VehicleControl(throttle=0.077, steering=0.25)
            
        return VehicleControl() 
        # print(self.time_counter)
        # if self.front_rgb_camera.data is not None:
        #     print("rgb: ", self.front_rgb_camera.data.shape)
        # if self.front_depth_camera.data is not None:
            # plt.imshow(self.front_depth_camera.data)
            # plt.show()
            # print(self.front_depth_camera.data)

        # return VehicleControl(throttle=0.077, steering=0.25)
        # print(trans)
        # # if self.time_counter < 500:
        # #     return VehicleControl(throttle=0.077, steering=0.25)
        # # else:
        # #     return VehicleControl(steering=0.25)
        # return VehicleControl()

        # if self.front_rgb_camera.data is not None:
        #     pos = self.vehicle.transform
        #     img = self.front_rgb_camera.data.copy()
        #     cv2.imshow(str(self.time_counter), img)
        #     cv2.waitKey(1)
        #     if self.detect_ball(img):
        #         #center ball and navigate slowly to ball
        #         return VehicleControl(throttle=0.05, steering=0)
        #     return VehicleControl()
        # else:
        #     return VehicleControl()

    def detect_ball(self, img):
        return False
        # low = (89, 30, 59)
        # actual = (114, 130, 57)
        # high = (194, 250, 15)
        # detected = (cv2.inRange(img, low, high).any())
        # print(detected)
        # return detected


