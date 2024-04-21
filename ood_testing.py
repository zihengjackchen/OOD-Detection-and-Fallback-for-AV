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



def demo(plot = False, pkl_dest = "/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00100/sensor_data/300.pkl"):
    with open("/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00100/sensor_data/300.pkl", "rb") as f:
        input_data = pkl.load(f)

    raw_rgb_dict = { key: input_data[key][1] for key in ['rgb', 'rgb_left', 'rgb_right'] }
    shady_image = { key: shadow.add_shadow(image=raw_rgb_dict[key], degree_of_shade=0.5) for key in raw_rgb_dict }
    rainy_image = { key: rain.add_rain(image=raw_rgb_dict[key], intensity=1500) for key in raw_rgb_dict }
    hazy_image = { key: haze.add_fog_random(image=raw_rgb_dict[key], reality = 150) for key in raw_rgb_dict}

    if plot:
        raw_rgb_dict_vis = np.concatenate(tuple([raw_rgb_dict[key] for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        shady_image_vis = np.concatenate(tuple([shadow.add_shadow(image = raw_rgb_dict[key], degree_of_shade = 0.5) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        rainy_image_vis = np.concatenate(tuple([rain.add_rain(image = raw_rgb_dict[key], intensity = 250) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        hazy_image_vis = np.concatenate(tuple([haze.add_fog_random(image = raw_rgb_dict[key], reality = 150) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)

        img = Image.fromarray(cv2.cvtColor(raw_rgb_dict_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/orig.png")

        img = Image.fromarray(cv2.cvtColor(shady_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/shady.png")

        img = Image.fromarray(cv2.cvtColor(rainy_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/rainy.png")

        img = Image.fromarray(cv2.cvtColor(hazy_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/hazy.png")

    print(maha.is_in_dist(raw_rgb_dict), maha.is_in_dist(shady_image), maha.is_in_dist(rainy_image), maha.is_in_dist(hazy_image))


def test_all_weather_on_single_frame(plot = False, pkl_dest = "/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00100/sensor_data/300.pkl"):
    with open(pkl_dest, "rb") as f:
        input_data = pkl.load(f)
    
    headers = ["effect_name", "parameter", "in_distribution"]
    filename = "/media/sheng/data4/projects/OOD/OOD-Detection-maha/testing/single_frame_OOD_detection.csv"

    with open(filename, mode = 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    raw_rgb_dict = { key: input_data[key][1] for key in ['rgb', 'rgb_left', 'rgb_right'] }
    
            
    for shade in range(1, 51):
        s = shade * 0.025
        shady_image = { key: shadow.add_shadow(image=raw_rgb_dict[key], degree_of_shade=s) for key in raw_rgb_dict }
        res = maha.is_in_dist(shady_image)
        with open(filename, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["shade", shade, res])
    
    for r in range(100, 2650, 50):
        rainy_image = { key: rain.add_rain(image=raw_rgb_dict[key], intensity = r) for key in raw_rgb_dict }
        res = maha.is_in_dist(rainy_image)
        with open(filename, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["rain", r, res])
    
    for h in range(10, 265, 5):
        hazy_image = { key: haze.add_fog_random(image=raw_rgb_dict[key], reality = h) for key in raw_rgb_dict}
        res = maha.is_in_dist(hazy_image)
        with open(filename, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["haze", h, res])
    
    if plot:
        raw_rgb_dict_vis = np.concatenate(tuple([raw_rgb_dict[key] for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        shady_image_vis = np.concatenate(tuple([shadow.add_shadow(image = raw_rgb_dict[key], degree_of_shade = 0.5) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        rainy_image_vis = np.concatenate(tuple([rain.add_rain(image = raw_rgb_dict[key], intensity = 250) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)
        hazy_image_vis = np.concatenate(tuple([haze.add_fog_random(image = raw_rgb_dict[key], reality = 150) for key in ['rgb_left', 'rgb', 'rgb_right']]), -2)

        img = Image.fromarray(cv2.cvtColor(raw_rgb_dict_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/orig.png")

        img = Image.fromarray(cv2.cvtColor(shady_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/shady.png")

        img = Image.fromarray(cv2.cvtColor(rainy_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/rainy.png")

        img = Image.fromarray(cv2.cvtColor(hazy_image_vis,cv2.COLOR_BGRA2RGB))
        img.save("/media/sheng/data4/projects/OOD/OOD-Detection-maha/demo_images/hazy.png")


def test_all_orig_frames(base_dir):
    
    all_paths = []
    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        # Check if current directory has the required subfolder structure
        if 'sensor_data' in dirs:
            sensor_data_path = os.path.join(root, 'sensor_data')
            # Iterate over each file in the sensor_data directory
            for filename in os.listdir(sensor_data_path):
                # Check for pickle files
                if filename.endswith('.pkl'):
                    file_path = os.path.join(sensor_data_path, filename)
                    all_paths.append(file_path)
    
    
    headers = ["file_path", "in_distribution"]
    filename = "/media/sheng/data4/projects/OOD/OOD-Detection-maha/testing/all_orig_frames_detection.csv"
    
    for pkl_dest in all_paths:
        with open(pkl_dest, "rb") as f:
            input_data = pkl.load(f)
        

        with open(filename, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        
        raw_rgb_dict = { key: input_data[key][1] for key in ['rgb', 'rgb_left', 'rgb_right'] }
        
        res = maha.is_in_dist(raw_rgb_dict)
        with open(filename, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pkl_dest, res])


if __name__ == "__main__":
    # Test if OOD can be detected on all ranges
    # test_all_weather_on_single_frame()
    
    # Test if all orig frames are in-dist
    test_all_orig_frames("/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source")