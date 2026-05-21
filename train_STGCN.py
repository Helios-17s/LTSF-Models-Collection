
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# ==========================================================
# 【STGCN 核心参数配置区域】 - 修改此处即可适配新数据
# ==========================================================

# 1. 路径与任务配置
DATA_PATH = "/content/dengue_processed.csv" # 数据文件路径 (CSV格式)
MODEL_ID = "stgcn_full_train"             # 实验任务名称，用于标识日志

# 2. 时间序列维度配置
SEQ_LEN = 12        # 输入序列长度 (观察过去多少个时间步)
PRED_LEN = 1         # 预测未来长度 (预测未来多少个时间步)
NUM_NODES = 52       # 特征维度 (节点数/列数，不含时间列)

# 3. 训练超参数
EPOCHS = 10          # 训练总轮数
BATCH_SIZE = 8       # 批大小 (建议 4-16)
LEARNING_RATE = 0.001 # 学习率

# 4. 硬件配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================

def create_stgcn_dataset(data_path, seq_len, pred_len):
    """数据读取与窗口化处理"""
    df = pd.read_csv(data_path)
    # 假设第一列是时间，从第二列开始是数值特征
    data = df.iloc[:, 1:].values.astype('float32')
    
    # 简单标准化
    mean, std = data.mean(), data.std()
    data = (data - mean) / std
    
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pred_len])
    
    return torch.tensor(np.array(X)).unsqueeze(-1), torch.tensor(np.array(Y)).unsqueeze(-1), mean, std

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0))
        self.spatial = nn.Linear(num_nodes, num_nodes)
        self.temporal2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.temporal1(x))
        x = self.spatial(x)
        x = self.relu(self.temporal2(x))
        return self.batch_norm(x)

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, seq_len, out_len):
        super(STGCN, self).__init__()
        self.block1 = STConvBlock(in_channels, 32, num_nodes)
        self.block2 = STConvBlock(32, 64, num_nodes)
        self.fully = nn.Linear(64 * seq_len, out_len)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.block1(x)
        x = self.block2(x)
        B, C, T, N = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, N, -1)
        x = self.fully(x)
        return x.permute(0, 2, 1).unsqueeze(-1) # (B, PRED_LEN, N, 1)

def train():
    X, Y, mean, std = create_stgcn_dataset(DATA_PATH, SEQ_LEN, PRED_LEN)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = STGCN(NUM_NODES, 1, SEQ_LEN, PRED_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--- 启动 STGCN 训练任务: {MODEL_ID} (共 {EPOCHS} 轮) ---")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_mae = 0
        start_time = time.time()
        model.train()
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            
            loss = criterion(output, batch_y)
            mae = mae_metric(output, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mae += mae.item()
        
        avg_mse = epoch_loss / len(loader)
        avg_mae = epoch_mae / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f} | Time: {time.time()-start_time:.2f}s")

    print("\n[Result] STGCN 10轮训练执行完毕。")
    print(f"最终指标汇总: MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")

if __name__ == '__main__':
    train()
