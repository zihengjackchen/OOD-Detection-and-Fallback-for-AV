from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import distance
from joblib import load

# Assuming you have a saved and fitted PCA model
pca = load('fitted_pca_left.joblib')  # Make sure the PCA model is loaded correctly

cov_matrix = np.load('left_covariance_matrix.npy')
mean_vector = np.load('left_mean_vector.npy')
inv_covariance_matrix = inv(cov_matrix)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    return df.values.flatten()  # Ensure this matches how your training data was preprocessed

# Load new data and transform
file_paths = ['left/image1.csv']
data = np.array([load_and_preprocess(fp) for fp in file_paths])


def image_distance(image, position):
    if position == 'right':
        pca = load('fitted_pca_right.joblib')  # Make sure the PCA model is loaded correctly
        cov_matrix = np.load('right_covariance_matrix.npy')
        mean_vector = np.load('right_mean_vector.npy')
        inv_covariance_matrix = inv(cov_matrix)
        data_pca_transformed = pca.transform(image.reshape(1, -1))  # Transform the data
        data_pca_transformed = np.append(data_pca_transformed, [0,0,0])
        #data_pca_transformed = np.delete(data_pca_transformed, slice(-7, None))
        maha_distance = distance.mahalanobis(data_pca_transformed.flatten(), mean_vector, inv_covariance_matrix)
        return maha_distance
    elif position == 'left':
        pca = load('fitted_pca_left.joblib')  # Make sure the PCA model is loaded correctly
        cov_matrix = np.load('left_covariance_matrix.npy')
        mean_vector = np.load('left_mean_vector.npy')
        inv_covariance_matrix = inv(cov_matrix)
        data_pca_transformed = pca.transform(image.reshape(1, -1))  # Transform the data
        data_pca_transformed = np.delete(data_pca_transformed, slice(-7, None))
        maha_distance = distance.mahalanobis(data_pca_transformed.flatten(), mean_vector, inv_covariance_matrix)
        return maha_distance
    else:
        pca = load('fitted_pca_middle.joblib')  # Make sure the PCA model is loaded correctly
        cov_matrix = np.load('middle_covariance_matrix.npy')
        mean_vector = np.load('middle_mean_vector.npy')
        inv_covariance_matrix = inv(cov_matrix)
        data_pca_transformed = pca.transform(image.reshape(1, -1))  # Transform the data
        data_pca_transformed = np.append(data_pca_transformed, [0,0,0,0])
        maha_distance = distance.mahalanobis(data_pca_transformed.flatten(), mean_vector, inv_covariance_matrix)
        return maha_distance

def calculate_OOD(super_wide_image):
    left, middle, right = np.split(super_wide_image, 3)
    left_distance = image_distance(left, "left")
    middle_distance = image_distance(middle, "middle")
    right_distance = image_distance(right, "right")

print(image_distance(data, "left"))


