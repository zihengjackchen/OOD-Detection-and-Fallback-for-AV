import torch
import numpy as np
import pandas as pd
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

def is_in_distribution(model, input_data, threshold):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 不需要计算梯度
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        reconstructed_output = model(input_tensor)
        mse = nn.functional.mse_loss(input_tensor, reconstructed_output)  # 计算均方误差
        print(mse.item())
        return mse.item() < threshold  # 如果MSE小于阈值，则返回True，否则返回False

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = Autoencoder().to(device)
load_model_weights(inference_model, 'autoencoder_mid_weights.pth')

# 从文件中加载数据
test_data = load_and_preprocess('middle/image1.csv')  # 替换 'test.csv' 为你的测试数据文件路径

# 使用推理函数进行推断
threshold = 0.01  # 设置阈值
is_in_dist = is_in_distribution(inference_model, test_data, threshold)
print(f"输入数据{'在' if is_in_dist else '不在'}分布中")
