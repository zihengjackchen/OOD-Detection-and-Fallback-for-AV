import numpy as np
import pandas as pd

# Generate sequential data
num_points = 256 * 144  # Total number of points

# Generate x coordinates (0-255)
x_values = np.tile(np.arange(256), 144)

# Generate sequential y coordinates (1-144)
y_values = np.repeat(np.arange(1, 145), 256)

# Generate random color values for R, G, B, A channels
r_values = np.random.randint(0, 256, num_points)
g_values = np.random.randint(0, 256, num_points)
b_values = np.random.randint(0, 256, num_points)
a_values = np.random.randint(0, 256, num_points)

# Create individual pandas Series for each column
x_series = pd.Series(x_values, name='X')
y_series = pd.Series(y_values, name='Y')
r_series = pd.Series(r_values, name='R')
g_series = pd.Series(g_values, name='G')
b_series = pd.Series(b_values, name='B')
a_series = pd.Series(a_values, name='A')

# Combine all Series into a single DataFrame
df = pd.concat([x_series, y_series, r_series, g_series, b_series, a_series], axis=1)

# Write DataFrame to CSV
df.to_csv('generated_data.csv', index=False)

print("Data saved to generated_data.csv")
