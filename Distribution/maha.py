import pandas as pd
import numpy as np

file_paths = ['right/image67.csv', 'right/image586.csv', 'right/image777.csv']

stacked_columns = []

# Loop through each file and process the data
for file_path in file_paths:
    print(file_path)
    data_list = []
    df = pd.read_csv(file_path)
    data_list.append(df[['R', 'G', 'B', 'A']].values)
    data_array = np.concatenate(data_list, axis=0).flatten()
    stacked_columns.append(data_array)

# Combine the processed data
result = np.column_stack(stacked_columns)
print(result)
print("Shape of result:", result.shape)

# Compute the mean and variance for each row
mean_vector = np.mean(result, axis=1)
variance_vector = np.var(result, axis=1)

# Create a DataFrame to store the mean and variance
df_stats = pd.DataFrame({
    'Mean': mean_vector,
    'Variance': variance_vector
})

# Save the DataFrame to a .csv file
df_stats.to_csv('stats.csv', index=False)

print("Mean Vector:", mean_vector)
print("Variance Vector:", variance_vector)
print("Data saved to stats.csv")
