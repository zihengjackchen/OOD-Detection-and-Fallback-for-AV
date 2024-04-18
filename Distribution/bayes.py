import os
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process image folder path.')
parser.add_argument('-left', action='store_true', help='Use left folder path.')
parser.add_argument('-right', action='store_true', help='Use right folder path.')
parser.add_argument('-middle', action='store_true', help='Use middle folder path.')
args = parser.parse_args()

# Determine folder path based on command line arguments
if args.left:
    folder_path = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/left'
elif args.right:
    folder_path = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/right'
elif args.middle:
    folder_path = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/middle'
else:
    print('Please specify folder path using -left, -right, or -middle.')
    exit()

# Read all CSV files
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store all data
all_data_list = []

count = 0
# Read all CSV files and concatenate data
for file in all_files:
    data = pd.read_csv(file)
    all_data_list.append(data)
    count += 1
    if count == 2:
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

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(flattened_pixels_array)

# Create target variable (assuming all images belong to "in distribution")
y = np.ones(flattened_pixels_array.shape[0])
print(len(y))

# Build and fit the model
clf = GaussianNB()
clf.fit(X_scaled, y)

# Save the model to a file
if args.left:
    joblib.dump(clf, 'naive_bayes_model_left.pkl')
elif args.right:
    joblib.dump(clf, 'naive_bayes_model_right.pkl')
elif args.middle:
    joblib.dump(clf, 'naive_bayes_model_middle.pkl')
