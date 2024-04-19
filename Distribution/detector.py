from PIL import Image
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import argparse

def get_image_pixels(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        pixels = [(x, y, *img.getpixel((x, y))) for x in range(width) for y in range(height)]
        return width, height, pixels

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Predict if an image is in distribution.')
    parser.add_argument('-image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-model_path', type=str, required=True, help='Path to the trained model.')
    

    args = parser.parse_args()

    # Load the trained model
    loaded_model = joblib.load(args.model_path)

    # Get pixel data from the input image
    width, height, pixel_data = get_image_pixels(args.image_path)

    # Extract features (x, y, R, G, B, A) from pixel data
    features = [[x, y, R, G, B, A] for x, y, R, G, B, A in pixel_data]
    #features=[[0, 0, 0, 0, 0, 0]]
    print(features)
    # Standardize the features using the same scaler from the training data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    # Predict whether the new image belongs to "in distribution"
    predictions = loaded_model.predict_proba(features_scaled)
    print(predictions)
    #probabilities = clf.predict_proba(X_new)
    #print(predictions)
    # Check the predictions
    if 1 in predictions:
        print("The new image belongs to 'in distribution'.")
    else:
        print("The new image does not belong to 'in distribution'.")
