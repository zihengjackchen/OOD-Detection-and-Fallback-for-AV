import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.stats import chi2
from scipy.linalg import inv, LinAlgError
from joblib import dump, load
import os
import pickle
from sklearn.preprocessing import StandardScaler

def load_pickle_files(base_dir):
    raw_images = {"rgb": [], "rgb_left": [], "rgb_right": []}
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
                                raw_images[key].append(np.array(input_data[key][1]).flatten())
                    
                    except Exception as e:
                        print(f"Failed to load data from {file_path}: {e}")

    return raw_images



if __name__ == "__main__":
    base_directory = "/media/sheng/data4/projects/OOD/OOD-Detection-maha/camera_input_source"
    left_images, right_images, front_images = load_pickle_files(base_directory)
    
    images = load_pickle_files(base_directory)
    
    for key in images:
        images[key] = np.array(images[key])

        # Step 1: Normalize the data
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(images[key])
        dump(scaler, f'/media/sheng/data4/projects/OOD/OOD-Detection-maha/Distribution/pca_saved/scalar_{key}.joblib')

        # Step 2: Apply PCA
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        data_transformed = pca.fit_transform(data_normalized)

        data_mean = np.mean(data_transformed, axis=0)
        data_covariance = np.cov(data_transformed, rowvar=False)

        # Save the fitted PCA model
        dump(pca, f'/media/sheng/data4/projects/OOD/OOD-Detection-maha/Distribution/pca_saved/fitted_pca_{key}.joblib')

        np.save(f'/media/sheng/data4/projects/OOD/OOD-Detection-maha/Distribution/pca_saved/data_covariance_{key}.npy', data_covariance)
        np.save(f'/media/sheng/data4/projects/OOD/OOD-Detection-maha/Distribution/pca_saved/data_mean_{key}.npy', data_mean)
