from PIL import Image
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
loaded_model = joblib.load('naive_bayes_model_right.pkl')

# Assuming you have a function or method to get pixel data from the new image
def get_image_pixels(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        pixels = [(x, y, *img.getpixel((x, y))) for x in range(width) for y in range(height)]
        return width, height, pixels

# Get pixel data from the new image
width, height, pixel_data = get_image_pixels('/mnt/shared/home/weihang6/OOD-Detection/OOD-Detection/middle_images/image64.png')

# Extract features (x, y, R, G, B, A) from pixel data
features = [[x, y, R, G, B, A] for x, y, R, G, B, A in pixel_data]

# Standardize the features using the same scaler from the training data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
#print(features_scaled)
# Predict whether the new image belongs to "in distribution"
predictions = loaded_model.predict(features_scaled)

# Check the predictions
if 1 in predictions:
    print("The new image belongs to 'in distribution'.")
else:
    print("The new image does not belong to 'in distribution'.")
