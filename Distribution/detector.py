from PIL import Image
import numpy as np
import pandas as pd

def image_to_dataframe(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Convert the image to RGBA mode (includes R, G, B, and A channels)
    img_rgba = img.convert("RGBA")
    
    # Get the width and height of the image
    width, height = img_rgba.size
    
    # Get the image data
    img_data = np.array(img_rgba)
    
    # Create an empty DataFrame to store pixel data
    df = pd.DataFrame(columns=['x', 'y', 'R', 'G', 'B', 'A'])
    
    # Traverse through each pixel of the image and append its coordinates and RGBA values to the DataFrame
    pixel_data = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = img_data[y, x]
            pixel_data.append({'x': x, 'y': y, 'R': r, 'G': g, 'B': b, 'A': a})
    
    df = pd.DataFrame(pixel_data)
    
    return df

# Image path
image_path = "test.png"

# Convert image data to DataFrame
df = image_to_dataframe(image_path)

# Save DataFrame to a .csv file
df.to_csv('image_pixels.csv', index=False)

print("Data saved to image_pixels.csv")

stacked_columns=[]
data_list = []
df = pd.read_csv('image_pixels.csv')
data_list.append(df[['R', 'G', 'B', 'A']].values)
data_array = np.concatenate(data_list, axis=0).flatten()
stacked_columns.append(data_array)
result = np.column_stack(stacked_columns)
#print(result)

# Load the stats.csv file
df_stats = pd.read_csv('stats.csv')

# Extract mean and variance from stats.csv
mean_vector = df_stats['Mean'].values
variance_vector = df_stats['Variance'].values

# Define the sampling interval
n = 1000# Calculate the offset for each nth value in result compared to its corresponding mean
# Calculate the offset for each nth value in result compared to its corresponding mean
# Calculate the offset for each nth value in result compared to its corresponding mean
offset = result[::n] - mean_vector[::n]

# Calculate relative offset, handling division by zero
relative_offset = np.zeros_like(offset)

# Find indices where variance is not zero
non_zero_indices = np.where(variance_vector[::n] != 0)[0]
print(variance_vector[::n][non_zero_indices])


# Calculate relative offset where variance is not zero

