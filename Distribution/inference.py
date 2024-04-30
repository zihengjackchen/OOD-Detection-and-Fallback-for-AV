import torch
from joblib import dump, load
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset,ConcatDataset
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    normalized_data = df.copy()
    normalized_data.iloc[:, 2:5] = df.iloc[:, 2:5] / 255.0  # Normalize RGB values
    return normalized_data.values  

def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))

def is_in_distribution(model, input_data, threshold,mean_mse):
    model.eval()  
    with torch.no_grad():  
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        reconstructed_output = model(input_tensor)
        mse = nn.functional.mse_loss(input_tensor, reconstructed_output) 
        mse_item=abs(mse.item()-mean_mse)
        return mse_item < threshold  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = Autoencoder().to(device)
load_model_weights(inference_model, 'autocode_saved/autoencoder_mid_weights.pth')

test_data = load_and_preprocess('middle/image873.csv')  


threshold = 0.01
mean_mse, variance_mse = load('autocode_saved/mse_mid_statistics.joblib')
is_in_dist = is_in_distribution(inference_model, test_data, threshold,mean_mse)
print(f"input{' is ' if is_in_dist else 'not '}in distribution")
