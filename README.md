# OOD-Detection
Interpreting Camera Input

Every scenario (e.g., `SINGLE_AGENT_fi_lead_slowdown_00100`) is run once, and the sensor data is collected for each scenario. Only 10 scenarios are provided as a sample, and more complete dataset is being generated currently. 

Data is connected 4 times per second. 25, 50, 75, 100 is what the sensors see at 0.25s, 0.5s, 0.75s, and 1.0s. There are a lot of data collected since typically a scneario runs for 25 seconds.

There is a `sensor_data` folder in each scenario containing the collected data. Each `.pkl` file can be opened in python using the `pickle` package. It contains more than just the camera data, so the other sensor data can be ignored for now.

Each `pkl` file stores the `input_data` variable. It is used as such:

```python
# Process raw sensor input
def tick(self, input_data):
    result = super().tick(input_data)
    result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

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


# Feed to LBC network to output an action
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

```

The demo images can be generated as such:
```python
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
```