import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset,ConcatDataset
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

# autoencoder model
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

# change dataset and number of image accordingly
file_paths = [f'left/image{i}.csv' for i in range(2)]
data = [torch.tensor(load_and_preprocess(fp), dtype=torch.float32) for fp in file_paths]

datasets = [TensorDataset(tensor) for tensor in data]

concat_dataset = ConcatDataset(datasets)
dataloader = DataLoader(concat_dataset, batch_size=1, shuffle=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for inputs in tqdm(dataloader):
        inputs = inputs[0].to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'autoencoder_left_weights.pth')