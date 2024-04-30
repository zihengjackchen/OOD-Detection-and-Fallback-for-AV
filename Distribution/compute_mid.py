import os
import pandas as pd

# Set the folder path
folder_path = 'middle'

# Store all MSE values
mse_values = []

# Iterate over all .csv files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        test_data = load_and_preprocess(file_path)  # Load the data
        mse = nn.functional.mse_loss(torch.tensor(test_data, dtype=torch.float32).to(device),
                                      inference_model(torch.tensor(test_data, dtype=torch.float32).to(device))).item()
        mse_values.append(mse)  # Compute and store the MSE value

# Compute the mean and variance
mean_mse = np.mean(mse_values)
variance_mse = np.var(mse_values)

# Create a DataFrame containing mean and variance
mse_df = pd.DataFrame({'Mean MSE': [mean_mse], 'Variance MSE': [variance_mse]})

# Save the DataFrame to a CSV file
mse_df.to_csv('mse_statistics.csv', index=False)

print("Mean and variance have been saved to 'mse_statistics.csv' file.")
