import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from joblib import Parallel, delayed

file_paths = ['right/image67.csv', 'right/image586.csv', 'right/image777.csv'] 


stacked_columns = []


for file_path in file_paths:
    print(file_path)
    data_list = []
    df = pd.read_csv(file_path)
    data_list.append(df[['R', 'G', 'B', 'A']].values)
    data_array=np.concatenate(data_list, axis=0).flatten()
    #print(data_list)
    #print(len(data_array))
    #data_array = np.concatenate(data_list, axis=0)
    #stacked_columns.append(data_array)
    
    stacked_columns.append(data_array)
result = np.column_stack(stacked_columns)
print(result)
print(len(result))
#should be 256*144*4 features
#result=data_array

mean_vector = np.mean(result, axis=1)
print("Mean Vector:", mean_vector)
#print(mean_vector)
#print((len(mean_vector)))
#covariance_matrix = np.cov(result, rowvar=True)
#print(covariance_matrix)

def compute_covariance(data):
    return np.cov(data, rowvar=False)

covariance_matrices = Parallel(n_jobs=-1)(delayed(compute_covariance)(data.T) for data in stacked_columns)

n_features = len(mean_vector)
merged_covariance_matrix = np.zeros((n_features, n_features))

for i, cov_matrix in enumerate(covariance_matrices):
    start_idx = i * n_features
    end_idx = (i + 1) * n_features
    merged_covariance_matrix[start_idx:end_idx, start_idx:end_idx] = cov_matrix

print("Merged Covariance Matrix:", merged_covariance_matrix)

merged_covariance_df = pd.DataFrame(merged_covariance_matrix)
merged_covariance_df.to_csv('merged_covariance_matrix.csv', index=False)

#covariance_df = pd.DataFrame(covariance_matrix)
#covariance_df.to_csv('covariance_matrix.csv', index=False)