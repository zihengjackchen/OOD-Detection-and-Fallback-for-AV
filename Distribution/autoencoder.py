import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset,ConcatDataset
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import pickle
from joblib import load


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(147456, 1024),  # Adjust these dimensions as needed
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 147456),  # Match input dimension
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_and_save_network(key, num_epochs = 50, base_dir = "/media/sheng/data4/projects/OOD/OOD-Detection/camera_input_source"):
    all_images = []
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
                    try:
                        with open(file_path, 'rb') as file:
                            # Load the pickle file
                            input_data = pickle.load(file)
                             
                            raw_img = np.array(input_data[key][1])/ 255.0
                            raw_img = raw_img.flatten()
                            
                            all_images.append(raw_img)
                    
                    except Exception as e:
                        print(f"Failed to load data from {file_path}: {e}")

    # # change dataset and number of image accordingly
    # file_paths = [f'middle/image{i}.csv' for i in range(2)]
    all_images = [torch.tensor(img, dtype=torch.float32) for img in all_images]
    data_tensor = torch.stack(all_images).float()  # Ensure all data is float and stacked into a single tensor

    dataset = TensorDataset(data_tensor)  # Create a dataset with the stacked tensor
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in tqdm(dataloader):
            inputs = inputs[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

    torch.save(model.state_dict(), f'/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/autoencoder_{key}_weights.pth')
        

def mse(array1, array2):
    # Ensure the arrays have the same shape
    assert array1.shape == array2.shape, "Arrays must have the same dimensions."
    # Compute MSE
    sq = (array1 - array2) ** 2
    
    return sq.mean()


def compute_and_save_stats(base_dir = "/media/sheng/data4/projects/OOD/OOD-Detection/camera_input_source", weights_path_folder = "/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/"):
    if os.path.exists("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/stats.pkl"):
        with open("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/stats.pkl", 'rb') as pkl_file:
            means, vars = pickle.load(pkl_file)
    else:
        count = 0
        
        all_distances = {"rgb": [], "rgb_left": [], "rgb_right": []}
        
        for key in ['rgb', 'rgb_left', 'rgb_right']:      
            weights_path = os.path.join(weights_path_folder, f"autoencoder_{key}_weights.pth")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            inference_model = Autoencoder().to(device)
            inference_model.load_state_dict(torch.load(weights_path))
            inference_model.eval()
        
            with torch.no_grad():              
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
                                try:
                                    with open(file_path, 'rb') as file:
                                        # Load the pickle file
                                        input_data = pickle.load(file)
                                                                
                                        raw_img = np.array(input_data[key][1])/ 255.0
                                        raw_img = raw_img.flatten()
                                        input_tensor = torch.tensor(raw_img, dtype=torch.float32).to(device)
                                        
                                        output_img = inference_model(input_tensor).cpu().numpy()
                                        distance = mse(raw_img, output_img) 
                                        all_distances[key].append(distance)
                                        
                                        count += 1
                                        print(count, count/972 * 100)
                                
                                except Exception as e:
                                    print(f"Failed to load data from {file_path}: {e}")

        
        means = {key: np.mean(all_distances[key]) for key in ['rgb', 'rgb_left', 'rgb_right']}
        vars = {key: np.var(all_distances[key]) for key in ['rgb', 'rgb_left', 'rgb_right']}
        
        print(all_distances, means, vars)
    
        with open("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/stats.pkl", 'wb') as pkl_file:
            pickle.dump([means, vars], pkl_file)
    
    return means, vars


def is_in_dist_test(data, key, threshold = 3):
    # Compute mean and variance for the specified key
    mean_value, variance_value = compute_and_save_stats()
    
    # Calculate absolute and relative deviation
    abs_deviation = np.abs(data - mean_value[key])
    relative_deviation = abs_deviation / np.sqrt(variance_value[key])
    
    print(f"abs_deviation: {abs_deviation}, relative_deviation: {relative_deviation}, threshold: {threshold}")
    if(relative_deviation > threshold):
        return False
    else:
        return True


def is_in_dist(raw_rgb_dict):
    diffs = {}
        
    for key in raw_rgb_dict:
        raw_img = raw_rgb_dict[key].flatten() / 255.0
        input_tensor = torch.tensor(raw_img, dtype=torch.float32)
        output_img = inference(input_tensor)
        
        distance = mse(input_tensor, output_img) 
        diffs[key] = distance
         
    OODs = [is_in_dist_test(diffs[key], key) for key in diffs] 
    
    # All 0 (OOD): False
    # If all three are out of distribution -> OOD
    # If any of the three images are in distribution -> in distribution
    return any(OODs)

def is_in_dist_batch(raw_rgb_dict_list, threshold = 3, weights_path_folder = "/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/"):
    res = []
    curr_img = []
        
    for key in ['rgb', 'rgb_left', 'rgb_right']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = os.path.join(weights_path_folder, f"autoencoder_{key}_weights.pth")
        inference_model = Autoencoder().to(device)
        inference_model.load_state_dict(torch.load(weights_path))
        inference_model.eval()

        dict_to_add = 0
        
        for raw_rgb_dict in raw_rgb_dict_list:
            
            if key == 'rgb':
                diffs = {}
                
                raw_img = raw_rgb_dict[key].flatten() / 255.0
                input_tensor = torch.tensor(raw_img, dtype=torch.float32).to(device)
                    
                with torch.no_grad():  
                    output = inference_model(input_tensor).to('cpu')
                
                distance = mse(input_tensor.cpu(), output) 
                diffs[key] = distance
            
                curr_img.append(diffs)
            
            else:                
                raw_img = raw_rgb_dict[key].flatten() / 255.0
                input_tensor = torch.tensor(raw_img, dtype=torch.float32).to(device)
                    
                with torch.no_grad():  
                    output = inference_model(input_tensor).to('cpu')
                
                distance = mse(input_tensor.cpu(), output) 
                curr_img[dict_to_add][key] = distance
            
                dict_to_add += 1
            
    for diffs in curr_img:
        curr_in_dist = []
        for key in diffs:
            mean_value, variance_value = compute_and_save_stats()
            # Calculate absolute and relative deviation
            abs_deviation = np.abs(distance - mean_value[key])
            relative_deviation = abs_deviation / np.sqrt(variance_value[key])
            
            if(relative_deviation > threshold):
                curr_in_dist.append(False)
            else:
                curr_in_dist.append(True)
        res.append(any(curr_in_dist))
    
    return res

def inference(img, weights_path = "/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/auto_saved/autoencoder_rgb_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    inference_model = Autoencoder().to(device)
    inference_model.load_state_dict(torch.load(weights_path))
    inference_model.eval()
    
    with torch.no_grad():  
        input_tensor = torch.tensor(img.flatten(), dtype=torch.float32).to(device)
        output = inference_model(input_tensor)
    
    return output.to('cpu')
    


if __name__ == "__main__":
    # Test if OOD can be detected on all ranges
    # test_all_weather_on_single_frame()
    
    # Test if all orig frames are in-dist
    # test_all_orig_frames("/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source")
    
    # train_and_save_network("rgb_right", num_epochs=20)
    
    compute_and_save_stats()