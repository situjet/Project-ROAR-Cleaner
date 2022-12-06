#return to base, model predictive control

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

class RTBAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
    
    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        trans = self.vehicle.transform
        depth_img = self.front_depth_camera.data
        start_trans = [0, 0] #replace this with the transform of the start/base positions

        x_t = trans.location.x
        z_t = trans.location.z
        psi = trans.rotation.yaw

        goal_pos = np.array([1, 1])
        goal_x = goal_pos[0]
        goal_y = goal_pos[1]

        if trans and depth_img and start_trans is not None:
            depth_img = depth_img.copy()
            depth_img[0:len(depth_img)//2] = 0

            #generate vector from vehicle to home/base positions
            home_vector = np.array([0 - x_t, 0 - z_t])
            goal_vec = np.array([goal_x - x_t, goal_y - z_t])

            #finding the deflection between psi and theta (optimal vector to go home)
            dot_prod_normalized = np.dot(goal_vec, home_vector)/(np.linalg.norm(goal_vec)*np.linalg.norm(home_vector))
            home_theta_tot = np.arccos(np.clip(dot_prod_normalized, -1.0, 1.0))
            #deflection stored as theta_err
            car_theta = home_theta_tot - psi
            theta_err = car_theta

            if np.abs(theta_err) > 30:

                #scan the depth along this vector
                FOV = 69.39
                center_u = 72

                #locates the u_ball that corresponds with horizon
                u_ball, t_ball = np.unravel_index(depth_img.argmax(), depth_img.shape)
                #this is the inverse of the p_agent control code
                v_ball = int((theta_err + center_u)/(FOV/center_u))
                
                dist = depth_img[u_ball, v_ball]
                error = 0

                if dist > 0.15:
                    error = np.array([1/10, theta_err/180])
                else:
                    #PID exploration controller
                    v_ball = t_ball
                    theta = (FOV/center_u) * v_ball - center_u
                    error = np.array([1/10, theta/180])

                #final actuation
                kp = np.array([[0.15, 0], [0, 1]])

                t, s = kp@error

                print(f'theta: {theta_err}, dist: {dist}, u: {u_ball}, v: {v_ball}')
                print(f'throttle: {t}, steering: {s}\n')
                return VehicleControl(throttle=t, steering=s+0.25)

            else:
                return VehicleControl(steering = .35) #turn right by a bit

        return VehicleControl(steering = 0.25)