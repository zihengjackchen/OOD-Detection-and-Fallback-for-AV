#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface
import cv2
from multipledispatch import dispatch
import numpy as np

import os
import pickle




class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, agent_id=0, duplicate=False, preprocessing=None, preparam=[]):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file)

        self.wallclock_t0 = None

        # for dual agent setup, duplicate mode run the agent anyway regardless of the sensor ID
        self.agent_id = agent_id
        self.duplicate = duplicate
        self.preprocessing = preprocessing
        self.preparam = preparam

        if self.preprocessing and self.agent_id == 1:
            print("enable image preprocessing for agent 1, preprocessing mode {}, parameters {}".format(self.preprocessing, self.preparam))

        if duplicate:
            print("Ignoring sensor data ID, agent will run at each timestep anyway.")

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    # propressing functions
    def _add_gaussian_noise(self, image, mean, std):
        noisy_image = np.random.normal(mean, std, image.shape) + image / 255
        noisy_image = np.clip(noisy_image, 0, 1)
        return (noisy_image * 255).astype(np.uint8)
    
    def _reduce_brightness(self, image, value):
        value = np.clip(value, 0, 255)
        value = np.uint8(value)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v[v < value] = 0
        v[v >= value] -= value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img.astype(np.int16)

    def _reduce_brightness_uniform(self, image, percentage):
        percentage /= 100
        percentage = np.clip(percentage, 0, 1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = v.astype(np.float)
        v = v * (1-percentage)
        v = v.astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img.astype(np.int16)

    def __call2__(self, policy_action):
        """
        Execute the agent call, e.g. agent(), but this time takes in a parameter action
        Returns the next vehicle controls and apply it
        """
        if (self.sensor_interface.get_current_queue_index() != self.agent_id) and not self.duplicate:
            return None, None

        # duplicate input data for duplicated agent setup
        if self.agent_id == 0 and self.duplicate:
            self.sensor_interface.duplicate_queue_data()

        # get in the input data (no need as we have already processed the data)
        # input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        control, dump_dict = self.run_step(policy_action, timestamp)
        control.manual_gear_shift = False

        return control, dump_dict

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        if (self.sensor_interface.get_current_queue_index() != self.agent_id) and not self.duplicate:
            return None, None

        # duplicate input data for duplicated agent setup
        if self.agent_id == 0 and self.duplicate:
            self.sensor_interface.duplicate_queue_data()
        
        # get in the input data
        input_data = self.sensor_interface.get_data()

        # Saving sensor info to pkl
        timestamp = int(GameTime.get_time() * 100) // 25 * 25
        main_folder = "/media/sheng/data4/projects/DiverseEnv/auto/paramsweep_results_lead_slowdown_test"
        # Ensure the base path exists
        if not os.path.exists(main_folder):
            raise ValueError("Provided base path does not exist.")
        # List all subdirectories in the base path
        subdirectories = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
        # Sort subdirectories and select the one with the largest name lexicographically
        if not subdirectories:
            raise ValueError("No subdirectories found in the provided base path.")
        largest_subdirectory = sorted(subdirectories)[-1]
        # Path to the largest subdirectory
        largest_subdirectory_path = os.path.join(main_folder, largest_subdirectory)
        # Create a new subfolder inside the largest subdirectory
        new_subfolder_path = os.path.join(largest_subdirectory_path, "sensor_data")
        os.makedirs(new_subfolder_path, exist_ok=True)  # Use exist_ok=True to avoid error if the folder already exists
        # Path for the .pkl file to save the variable
        pkl_file_path = os.path.join(new_subfolder_path, f"{timestamp}.pkl")

        if not os.path.exists(pkl_file_path):
            # Save the variable to a .pkl file
            with open(pkl_file_path, 'wb') as pkl_file:
                pickle.dump(input_data, pkl_file)
            print(f"Variable saved in {pkl_file_path}")





        # depends on the flag we might do preprocessing
        if self.preprocessing and self.agent_id == 1:
            if self.preprocessing == "gauss":
                for key in ["rgb", "rgb_left", "rgb_right"]:
                    data = input_data[key][1][:,:,0:3]
                    data = self._add_gaussian_noise(data, self.preparam[0], self.preparam[1])
                    input_data[key][1][:,:,0:3] = data
            
            elif self.preprocessing == "darken_abs":
                for key in ["rgb", "rgb_left", "rgb_right"]:
                    data = input_data[key][1][:,:,0:3]
                    data = self._reduce_brightness(data, self.preparam[0])
                    input_data[key][1][:,:,0:3] = data
            
            elif self.preprocessing == "darken_uniform":
                for key in ["rgb", "rgb_left", "rgb_right"]:
                    data = input_data[key][1][:,:,0:3]
                    data = self._reduce_brightness_uniform(data, self.preparam[0])
                    input_data[key][1][:,:,0:3] = data
            
            else:
                raise NotImplementedError("processing method {} is not implemented.".format(self.preprocessing))

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        # print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))

        control, dump_dict = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control, dump_dict

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
