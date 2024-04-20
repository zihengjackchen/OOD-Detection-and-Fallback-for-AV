from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import distance
from joblib import load

# Assuming you have a saved and fitted PCA model
pca = load('fitted_pca.joblib')  # Make sure the PCA model is loaded correctly

cov_matrix = np.load('right_covariance_matrix.npy')
mean_vector = np.load('right_mean_vector.npy')
inv_covariance_matrix = inv(cov_matrix)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    return df.values.flatten()  # Ensure this matches how your training data was preprocessed

# Load new data and transform
file_paths = ['generated_data.csv']
data = np.array([load_and_preprocess(fp) for fp in file_paths])
# Check for NaNs and fill them or drop them
if np.isnan(data).any():
    # Option 1: Fill NaNs with the mean of each column
    # data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)

    # Option 2: Drop rows with any NaNs
    data = data[~np.isnan(data).any(axis=1)]
data_pca_transformed = pca.transform(data.reshape(1, -1))  # Transform the data
data_pca_transformed = np.append(data_pca_transformed, [0,0,0])
# Compute Mahalanobis distance
maha_distance = distance.mahalanobis(data_pca_transformed.flatten(), mean_vector, inv_covariance_matrix)
print("Mahalanobis Distance:", maha_distance)

