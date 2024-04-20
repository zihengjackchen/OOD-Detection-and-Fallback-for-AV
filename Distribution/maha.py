from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import distance
from joblib import load
import os

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


def compute_stats(folder_path, category):
    all_distances = []
    
    # Get all .csv files in the folder
    all_files = os.listdir(folder_path)
    
    for filename in all_files:
        if filename.endswith('.csv'):
            # Build the full file path
            full_path = os.path.join(folder_path, filename)
            #print(full_path)
            
            # Load and preprocess the data
            data = load_and_preprocess(full_path)
            
            # Calculate image distance
            distance = image_distance(data, category)
            #print(distance)
            
            # Store the image distance
            all_distances.append(distance)
    
    # Calculate mean and variance
    mean_distance = np.mean(all_distances)
    var_distance = np.var(all_distances)
    
    return mean_distance, var_distance

def compute_and_save_stats():
    # Compute the mean and variance for the left, right, and middle categories
    left_mean, left_var = compute_stats('left', 'left')
    print(f"Left Mean: {left_mean}, Left Variance: {left_var}")

    right_mean, right_var = compute_stats('right', 'right')
    print(f"Right Mean: {right_mean}, Right Variance: {right_var}")

    middle_mean, middle_var = compute_stats('middle', 'middle')
    print(f"Middle Mean: {middle_mean}, Middle Variance: {middle_var}")

    # Create a DataFrame
    stats_df = pd.DataFrame({
        'Category': ['left', 'right', 'middle'],
        'Mean': [left_mean, right_mean, middle_mean],
        'Variance': [left_var, right_var, middle_var]
    })

    # Save the DataFrame to a CSV file
    stats_df.to_csv('stats.csv', index=False)

def load_mean_var(category_name, file_name='stats'):
    # Read the stats data
    df = pd.read_csv(f"{file_name}.csv")
    
    # Extract mean and variance for the specified category
    mean_value = df[df['Category'] == category_name]['Mean'].values[0]
    variance_value = df[df['Category'] == category_name]['Variance'].values[0]
    
    return mean_value, variance_value

def OOD_test(data, category):
    # Compute mean and variance for the specified category
    mean_value, variance_value = load_mean_var(category)
    
    # Calculate absolute and relative deviation
    abs_deviation = np.abs(data - mean_value)
    relative_deviation = abs_deviation / np.sqrt(variance_value)
    if(relative_deviation>2):
        return 0
    else:
        return 1


def calculate_OOD(super_wide_image):
    left, middle, right = np.split(super_wide_image, 3)
    left_distance = image_distance(left, "left")
    left_OOD=OOD_test(left_distance,"left")
    middle_distance = image_distance(middle, "middle")
    middle_OOD=OOD_test(middle_distance,"middle")
    right_distance = image_distance(right, "right")
    right_OOD=OOD_test(right_distance,"right")
    if(left_OOD==0 and middle_OOD==0 and right_OOD==0):
        print("Out of Distribution")
        return 0
    else:
        print("In Distribution")
        return 1

#compute_and_save_stats() uncomment to generate stats.csv
print(image_distance(data,"left"))
#calculate_OOD(super_wide_image)