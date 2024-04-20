import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.stats import chi2
from scipy.linalg import inv, LinAlgError
from joblib import dump, load

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # Assuming RGB to grayscale conversion or direct loading if already processed
    return df.values.flatten()  # Flatten if necessary, adjust based on your data's structure

file_paths = []
for i in range(972):
    file_paths.append(f'middle/image{i}.csv')
data = np.array([load_and_preprocess(fp) for fp in file_paths])

if np.isnan(data).any():
    data = data[~np.isnan(data).any(axis=1)]

data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)

# Avoid division by zero by setting zero variance features to zero
data_std[data_std == 0] = 1
data_normalized = (data - data_mean) / data_std

# Replace any resulting NaNs with 0 (which could still occur if there are NaNs in data)
data_normalized = np.nan_to_num(data_normalized)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
data_transformed = pca.fit_transform(data_normalized)
data_mean = np.mean(data_transformed, axis=0)
covariance_matrix = np.cov(data_transformed, rowvar=False)

pca.fit(data)

# Save the fitted PCA model
dump(pca, 'fitted_pca_middle.joblib')

np.save('middle_covariance_matrix.npy', covariance_matrix)
np.save('middle_mean_vector.npy', data_mean)
print("Covariance matrix saved to 'covariance_matrix.npy'.")