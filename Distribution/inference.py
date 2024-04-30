import torch
from joblib import dump, load
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset,ConcatDataset
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from autoencoder import Autoencoder, load_and_preprocess

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
