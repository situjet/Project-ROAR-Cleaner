from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.le_bozo.ball_detector import get_circles
from ROAR.le_bozo.gripper_client import *
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
import cv2

class PAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        
        trans = self.vehicle.transform
        rgb_img = self.front_rgb_camera.data
        depth_img = self.front_depth_camera.data

        # if self.time_counter % 3 == 0:
        #     commandGripper("close")

        # print("outside")
        if trans is not None and rgb_img is not None and depth_img is not None:
            # print("rgb present")
            # rgb_img   : (1280, 720, 3)
            # depth_img : (256, 144)

            x_t = trans.location.x
            z_t = trans.location.z
            psi = trans.rotation.yaw

            if self.time_counter % 1 == 0:
                circles, mask = get_circles(rgb_img)
                # circles = np.array([[[1, 2, 3]]])
                if circles is not None:
                    for i in range(len(circles)):
                        circles2 = np.round(circles[0, :]).astype("int")
                        cv2.circle(mask, center=(circles2[i, 0], circles2[i, 1]), radius=circles2[i, 2], color=(0, 255, 0), thickness=2)
                    
                    u_ball, v_ball = int(circles[0][0][1]//5), int(circles[0][0][0]//5)
                    FOV = 69.39
                    center_u = 72
                    offset = v_ball - center_u
                    theta = (FOV/center_u) * offset

                    dist = depth_img[u_ball][v_ball]
                    depth = abs(dist*np.cos(theta))
                    offset_h = dist*np.sin(theta)

                    # x_ball = x_t + (np.sin(psi)*depth) + (np.cos(psi)*offset_h)
                    # z_ball = z_t - (np.cos(psi)*depth) + (np.sin(psi)*offset_h)

                    x_ball = x_t + dist*np.sin(psi+theta)
                    z_ball = z_t - dist*np.cos(psi+theta)

                    error = np.array([depth, theta/180])
                    kp = np.array([[5, 0], [0, 1]])

                    t, s = kp@error

                    print(f'circles: {len(circles[0])}, theta: {theta}, dist: {dist}, depth: {depth}, u: {u_ball}, v: {v_ball}, x: {x_ball}, z: {z_ball}')
                    print(f'throttle: {t}, steering: {s}\n')

                    return VehicleControl(throttle=t, steering=s+0.25)
                else:
                    print('no ball')
                cv2.imshow("img", mask)

                # if circles is not None:

                #     print("ball found: ", circles)
            # else:
                # print("not found")

            # print(f'x_t: {x_t}, z_t: {z_t}, psi: {psi}')

            # sampled_rgb_img = cv2.resize(rgb_img, dsize=(256, 144), interpolation=cv2.INTER_CUBIC)

            # u_ball, v_ball = 70, 70
            # FOV = 70
            # center_u = 72
            # offset = u_ball - center_u
            # theta = (FOV/center_u) * offset
            # print(theta)

            # dist = depth_img[u_ball][v_ball]
            # depth = dist*np.cos(theta)

            # x_ball = x_t + dist*np.sin(psi+theta)
            # z_ball = z_t + dist*np.cos(psi+theta)

            # error = np.array([depth/10, theta/180])
            # kp = np.array([[.5, 0], [0, 0.5]])

            # throttle, steering = kp@error
            # print(f'throttle: {throttle}, steering: {steering}')
        
        return VehicleControl(steering=0.25)