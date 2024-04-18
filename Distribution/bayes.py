import os
import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 1. Read all CSV files
folder_path = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/right'
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store all data
all_data_list = []

count=0
# Read all CSV files and concatenate data
for file in all_files:
    data = pd.read_csv(file)
    all_data_list.append(data)
    count+=1
    if(count==2):
        break


# Concatenate all data into a single DataFrame
all_data = pd.concat(all_data_list, ignore_index=True)

# Extract pixel values as features
features = ['X', 'Y', 'R', 'G', 'B', 'A']
X = all_data[features]

# Initialize an empty list to store flattened pixel data
flattened_pixels = []

# Flatten pixel data for each image
for i in range(len(all_data_list)):
    image_data = all_data_list[i][features].values
    flattened_pixels.extend(image_data)

# Convert the flattened pixels list to a numpy array
flattened_pixels_array = np.array(flattened_pixels)
print(flattened_pixels_array)

# 2. Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(flattened_pixels_array)

# Create target variable (assuming all images belong to "in distribution")
y = np.ones(flattened_pixels_array.shape[0])
print(len(y))

# 3. Build and fit the model
clf = GaussianNB()
clf.fit(X_scaled, y)

# Save the model to a file
joblib.dump(clf, 'naive_bayes_model_right.pkl')
