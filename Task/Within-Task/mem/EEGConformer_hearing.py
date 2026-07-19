import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# =========================================================
# 1. LOAD DATA (直接从原始 MAT 矩阵加载时域信号)
# =========================================================
mat_path = "../../../记忆阶段/EEG_hearing_memory.mat"
var_name = "EEG_hearing"

print("=" * 60)
print(f"Loading raw EEG MAT file from: {mat_path}")
print("=" * 60)

mat_data = sio.loadmat(mat_path)
raw_matrix = mat_data[var_name]  # 形状: (22, 60, 1600, 64)

# 解析原始维度
S, C, T, N = raw_matrix.shape  # S=22被试, C=60通道, T=1600时间点, N=64试次
num_classes = S

# 将维度从 (S, C, T, N) 调换为 (S, N, C, T) 以便按样本提取
# 转换后形状: (22, 64, 60, 1600)
X_permuted = np.transpose(raw_matrix, (0, 3, 1, 2))

# 展平前两维，将数据重组为独立的样本 -> 形状: (1408, 60, 1600)
X = X_permuted.reshape(S * N, C, T)

# 构建对应的脑纹识别标签 (每个被试有 64 个 Trial)
y = np.repeat(np.arange(S), N)  # 形状: (1408,)

print(f"Successfully loaded MAT data.")
print(f"Total Samples: {len(X)}, Channels: {C}, TimePoints: {T}")
print(f"Total Subjects (Classes): {num_classes}")

# =========================================================
# 2. WITHIN-SUBJECT SPLIT (分层 Trial 划分)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================================
# 3. NORMALIZATION (时域 Z-score 标准化)
# =========================================================
N_train = X_train.shape[0]
N_test = X_test.shape[0]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(N_train, -1)).reshape(N_train, C, T)
X_test_scaled = scaler.transform(X_test.reshape(N_test, -1)).reshape(N_test, C, T)

# =========================================================
# 4. ADD CHANNEL DIM & CONVERT TO TENSOR
# =========================================================
# 符合 EEGConformer 标准输入格式 -> (N, 1, Channels, TimePoints)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# =========================================================
# 5. EEGConformer MODEL IMPLEMENTATION
# =========================================================
class EEGConformer(nn.Module):
    def __init__(self, num_classes, chans=60, times=1600, emb_size=40, depth=6, heads=8):
        super().__init__()

        # 1. 浅层一维时间卷积 (Temporal Component)
        # 输入: (Batch, 1, Chans, Times)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(emb_size),
            nn.ELU()
        )

        # 2. 空间深度卷积 (Spatial Component)
        # 输入: (Batch, emb_size, Chans, Times)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(chans, 1), stride=(1, 1), groups=emb_size),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),  # 降低时间分辨率
            nn.Dropout(0.2)
        )

        # 计算经过卷积池化后的时间步长
        # 1600 / 8 = 200
        reduced_times = times // 8

        # 3. Transformer 模块 (Transformer Component)
        # nn.TransformerEncoder 要求输入维度为 (Sequence_Length, Batch, Embedding_Size)
        # 在这里，时间长度 reduced_times 就是 Sequence_Length
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=emb_size * 2,
            dropout=0.3,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. 全连接分类头
        self.classifier = nn.Sequential(
            nn.Linear(emb_size * reduced_times, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. 卷积提取特征
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)  # 形状变为: [Batch, emb_size, 1, reduced_times]

        # 2. 维度调整以符合 Transformer 的输入要求
        x = x.squeeze(2)  # 形状变为: [Batch, emb_size, reduced_times]
        x = x.permute(2, 0, 1)  # 变形为: [reduced_times, Batch, emb_size]

        # 3. Transformer 编码
        x = self.transformer(x)  # 形状保持: [reduced_times, Batch, emb_size]

        # 4. 展平并分类
        x = x.permute(1, 0, 2)  # 回到: [Batch, reduced_times, emb_size]
        x = x.reshape(x.size(0), -1)  # 展平为二维: [Batch, reduced_times * emb_size]

        out = self.classifier(x)
        return out


# =========================================================
# 6. TRAIN
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化 EEGConformer
model = EEGConformer(num_classes=num_classes, chans=C, times=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 送入设备
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

batch_size = 32
epochs = 200

print("\n🚀 开始训练原始时域脑电 EEGConformer 模型...")
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(X_train_tensor))
    total_loss = 0

    for i in range(0, len(X_train_tensor), batch_size):
        idx = perm[i:i + batch_size]
        xb = X_train_tensor[idx]
        yb = y_train_tensor[idx]

        optimizer.zero_grad()
        out = model(xb)

        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 每个 Epoch 验证测试集准确率
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor).argmax(dim=1)
        acc = (pred == y_test_tensor).float().mean().item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:03d} | Loss {total_loss:.3f} | Test Acc {acc:.4f}")

# =========================================================
# 7. FINAL EVALUATION (计算 ACC, REC, F1, EER)
# =========================================================
print("\n正在执行最终评估指标计算...")
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

y_true = y_test_tensor.cpu().numpy()

# 1. 计算核心分类指标
final_acc = accuracy_score(y_true, preds)
final_rec = recall_score(y_true, preds, average='macro')
final_f1 = f1_score(y_true, preds, average='macro')

# 2. 计算封闭集 1-vs-All 统一等错误率 (EER)
y_true_onehot = np.eye(num_classes)[y_true]
y_true_flat = y_true_onehot.ravel()
y_pred_flat = probs.ravel()

fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print("\n" + "=" * 30)
print("       TEST SET METRICS (EEGConformer)")
print("=" * 30)
print(f"Accuracy  (ACC) : {final_acc:.4f}")
print(f"Recall    (REC) : {final_rec:.4f}")
print(f"F1-Score  (F1)  : {final_f1:.4f}")
print(f"Equal Error Rate (EER): {eer:.4f} ({eer * 100:.2f}%)")
print("=" * 30)

# =========================================================
# 8. 模型保存
# =========================================================
model_path = "mem_raw_eeg_conformer_deaf.pth"
torch.save(model.state_dict(), model_path)
print(f"✔ Model weights successfully saved to: '{model_path}'")
print("=" * 30)