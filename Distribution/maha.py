from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import distance
from joblib import load
import os
import pickle


def pre_normalize(vector, key):
    scaler = load(f'/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/scalar_{key}.joblib')
    vector = vector.reshape(1, -1)
    return scaler.transform(vector)


def image_distance(image, key):
    pca = load(f'/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/fitted_pca_{key}.joblib')  # Make sure the PCA model is loaded correctly
    data_covariance = np.load(f'/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/data_covariance_{key}.npy')
    data_mean = np.load(f'/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/data_mean_{key}.npy')
    inv_data_covariance = inv(data_covariance)
    data_pca_transformed = pca.transform(image)  # Transform the data
    # data_pca_transformed = np.append(data_pca_transformed, [0,0,0])
    # #data_pca_transformed = np.delete(data_pca_transformed, slice(-7, None))
    maha_distance = distance.mahalanobis(data_pca_transformed.flatten(), data_mean, inv_data_covariance)
    return maha_distance


def compute_and_save_stats(base_dir = "/media/sheng/data4/projects/OOD/OOD-Detection/camera_input_source"):
    if os.path.exists("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/stats.pkl"):
        with open("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/stats.pkl", 'rb') as pkl_file:
            means, vars = pickle.load(pkl_file)
    else:
        all_distances = {"rgb": [], "rgb_left": [], "rgb_right": []}
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
                                
                                for key in ['rgb', 'rgb_left', 'rgb_right']:
                                    # Calculate image distance
                                    
                                    raw_img = np.array(input_data[key][1]).flatten()
                                    raw_img = pre_normalize(raw_img, key)
                                    
                                    distance = image_distance(raw_img, key)
                                    all_distances[key].append(distance)
                        
                        except Exception as e:
                            print(f"Failed to load data from {file_path}: {e}")

        means = {key: np.mean(all_distances[key]) for key in all_distances}
        vars = {key: np.var(all_distances[key]) for key in all_distances}
    
        with open("/media/sheng/data4/projects/OOD/OOD-Detection/Distribution/pca_saved/stats.pkl", 'wb') as pkl_file:
            pickle.dump([means, vars], pkl_file)
    
    return means, vars



def OOD_test(data, category, threshold = 3):
    # Compute mean and variance for the specified category
    mean_value, variance_value = compute_and_save_stats()
    
    # Calculate absolute and relative deviation
    abs_deviation = np.abs(data - mean_value[category])
    relative_deviation = abs_deviation / np.sqrt(variance_value[category])
    
    if(relative_deviation > threshold):
        return 0
    else:
        return 1


def is_in_dist(raw_rgb_dict):
    distances = {key: image_distance(pre_normalize(raw_rgb_dict[key], key), key) for key in raw_rgb_dict}
    OODs = [OOD_test(distances[key], key) for key in distances] 
    
    # All 0 (OOD): False
    return any(OODs)


