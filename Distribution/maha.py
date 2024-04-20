from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import distance
from joblib import load

# Assuming the model was previously fitted and saved
pca = load('fitted_pca.joblib')

cov_matrix = np.load('right_covariance_matrix.npy')
mean_vector = np.load('right_mean_vector.npy')
inv_covariance_matrix = inv(cov_matrix)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    return df.values.flatten()  # Ensure this matches how your training data was preprocessed

# Load new data and transform
new_image = pd.read_csv('right/image0.csv')[['R', 'G', 'B', 'A']].values.ravel()
new_features = PCA(n_components=50).fit_transform(new_image.reshape(1, -1))

# Check dimensions before computing the distance
print("Data dimensions:", new_features.flatten().shape)
print("Mean vector dimensions:", mean_vector.shape)

# Compute Mahalanobis distance
maha_distance = distance.mahalanobis(new_features.flatten(), mean_vector, inv_covariance_matrix)
print("Mahalanobis Distance:", maha_distance)
