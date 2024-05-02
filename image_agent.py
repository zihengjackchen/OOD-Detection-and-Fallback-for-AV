import os
import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter


from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController
import pickle as pkl

from image_noise_modification import cloud_shadow_effect_adder as shadow
from image_noise_modification import haze_effect_adder as haze
from image_noise_modification import rain_effect_adder as rain

from Distribution import maha
import numpy as np

from PIL import Image, ImageDraw
import cv2

import csv
import os

# DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
DEBUG = True

def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step, timestamp):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))

    # Save camera input for debugging
    # print_time = int(timestamp * 100) // 25 * 25
    # main_folder = "/media/sheng/data4/projects/DiverseEnv/auto/paramsweep_results_lead_slowdown_test"
    # # Ensure the base path exists
    # if not os.path.exists(main_folder):
    #     raise ValueError("Provided base path does not exist.")
    # if not os.path.exists(f"{main_folder}/{print_time}.png"):
    #     cv2.imwrite(f"{main_folder}/{print_time}.png", cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    #     print(os.getcwd())

    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()
        ##-- pyTorch FI Integration--##
        # self.net.init_pytorch_fi()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.5, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=0.5, K_I=0.25, K_D=0.06, n=40)

    def tick(self, input_data):
        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        raw_rgb_dict = {key: input_data[key][1] for key in ['rgb', 'rgb_left', 'rgb_right']}

        with open(
                "/media/sheng/data4/projects/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/team_code/orig_combs.pkl",
                "rb") as f:
            weather, param = pkl.load(f)

        if weather == "shade":
            image_for_ood = {key: shadow.add_shadow(image=raw_rgb_dict[key], degree_of_shade=param) for key in raw_rgb_dict}
            weather_img_concat = []
            for x in ['rgb', 'rgb_left', 'rgb_right']:
                param *= 0.15
                weather_img = shadow.add_shadow(image=input_data[x][1], degree_of_shade=param)
                weather_img_cvt = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)
                weather_img_concat.append(weather_img_cvt)
                result[x] = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)
            result['image'] = np.concatenate(tuple(weather_img_concat), -1)
            print(f"Injected shade with param {param}")


        elif weather == "rain":
            param *= 150
            image_for_ood = {key: rain.add_rain(image=raw_rgb_dict[key], intensity=param) for key in raw_rgb_dict}
            weather_img_concat = []
            for x in ['rgb', 'rgb_left', 'rgb_right']:
                weather_img = rain.add_rain(image=input_data[x][1], intensity=param)
                weather_img_cvt = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)
                weather_img_concat.append(weather_img_cvt)
                result[x] = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)

            result['image'] = np.concatenate(tuple(weather_img_concat), -1)
            print(f"Injected rain with param {param}")


        elif weather == "haze":
            param *= 25
            image_for_ood = {key: haze.add_fog_random(image=raw_rgb_dict[key], reality=param) for key in raw_rgb_dict}
            weather_img_concat = []
            for x in ['rgb', 'rgb_left', 'rgb_right']:
                weather_img = haze.add_fog_random(image=input_data[x][1], reality=param)
                weather_img_cvt = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)
                weather_img_concat.append(weather_img_cvt)
                result[x] = cv2.cvtColor(weather_img[:, :, :3], cv2.COLOR_BGR2RGB)

            result['image'] = np.concatenate(tuple(weather_img_concat), -1)
            print(f"Injected haze with param {param}")


        print(f"Detected OOD: {maha.is_in_dist(image_for_ood)}")
        result["ood"] = maha.is_in_dist(image_for_ood)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        far_node, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target

        return result

    @torch.no_grad()
    def run_step_using_learned_controller(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        control = self.net.controller(points).cpu().squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if tick_data["ood"]:
            control.throttle *= 0.7
            control.brake = float(desired_speed < 0.4 * 1.3 or (speed / desired_speed) > 1.05)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step, timestamp)

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy()

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer_unclipped = steer
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 3.0
        # desired_speed *= (1 - abs(angle)) ** 2

        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.05

        delta = np.clip(desired_speed - speed, 0.0, 100.35)
        throttle = self._speed_controller.step(delta)
        throttle_unclipped = throttle
        throttle = np.clip(throttle, 0.0, 0.9)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step, timestamp)
        
        dump_dict = dict()
        dump_dict["points_cam"] = points_cam
        dump_dict["points_world"] = points_world
        dump_dict["steer_error"] = angle
        dump_dict["speed_error"] = delta
        dump_dict["steer_unclipped"] = steer_unclipped
        dump_dict["throttle_unclipped"] = throttle_unclipped
        dump_dict["desired_speed"] = desired_speed
        dump_dict["speed"] = speed
        return control, dump_dict

    ##-- pyTorch FI Integration--##
    def get_pfi_model(self):
        return self.net.get_pfi_model()

    def set_pfi_inj(self, pfi_inj, enable=True):
        self.net.pfi_inj = pfi_inj
        self.net.fi_enable=enable

    def enable_fi(enable=True):
        self.net.enable_fi(enable)
    ##-- pyTorch FI Integration End--##
