import numpy as np
import os
import pickle
from PIL import Image
folder_path = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00100/sensor_data'
folder_path1 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00101/sensor_data'
folder_path2 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00102/sensor_data'
folder_path3 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00103/sensor_data'
folder_path4 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00104/sensor_data'
folder_path5 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00105/sensor_data'
folder_path6 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00106/sensor_data'
folder_path7 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00107/sensor_data'
folder_path8 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00108/sensor_data'
folder_path9 = '../camera_input_source/SINGLE_AGENT_fi_lead_slowdown_00109/sensor_data'


files = os.listdir(folder_path)
files1 = os.listdir(folder_path1)
files2 = os.listdir(folder_path2)
files3 = os.listdir(folder_path3)
files4 = os.listdir(folder_path4)
files5 = os.listdir(folder_path5)
files6 = os.listdir(folder_path6)
files7 = os.listdir(folder_path7)
files8 = os.listdir(folder_path8)
files9 = os.listdir(folder_path9)

mega_list = files + files1 + files2 + files3 + files4 + files5 + files6 + files7 + files8 + files9
pkl_files = [file for file in mega_list if file.endswith('.pkl')]

pkl_data = []

for file_name in pkl_files:
    with open(os.path.join(folder_path, file_name), 'rb') as file:
        pkl_data.append(pickle.load(file)) 

rgb_left_image_list = []
rgb_image_list = []
rgb_right_image_list = []

for data in pkl_data:
    rgb_left_image_list.append(data['rgb_left'][1])
    rgb_image_list.append(data['rgb'][1])
    rgb_right_image_list.append(data['rgb_right'][1])

for i in range(len(rgb_image_list)):
    left_image = Image.fromarray(rgb_left_image_list[i])
    middle_image = Image.fromarray(rgb_image_list[i])
    right_image = Image.fromarray(rgb_right_image_list[i])

    left_image.save(f'left_images/image{i}.png')
    middle_image.save(f'middle_images/image{i}.png')
    right_image.save(f'right_images/image{i}.png')

    




