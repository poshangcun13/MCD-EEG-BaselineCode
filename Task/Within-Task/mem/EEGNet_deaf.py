import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# =========================================================
# 1. LOAD DATA (去掉任何滤波步骤，直接读取原始信号)
# =========================================================
mat_path = "../../../记忆阶段/EEG_deaf_memory.mat"
var_name = "EEG_deaf"

mat_data = sio.loadmat(mat_path)
raw_matrix = mat_data[var_name]  # (22, 60, 1600, 64)

S, C, T, N = raw_matrix.shape
num_classes = S

# 调整形状为 (Samples, 1, Channels, Time) -> 1 代表单频带(原始信号)
X = np.transpose(raw_matrix, (0, 3, 1, 2)).reshape(S * N, 1, C, T)
y = np.repeat(np.arange(S), N)

print("原始数据处理完成，输入形状:", X.shape)  # (1472, 1, 60, 1600)

# =========================================================
# 2. SPLIT & Z-SCORE (按通道独立归一化)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 针对原始信号，进行通道内归一化
for c in range(C):
    scaler = StandardScaler()
    # X_train[:, 0, c, :] 形状为 (Samples, Time)
    X_train[:, 0, c, :] = scaler.fit_transform(X_train[:, 0, c, :])
    X_test[:, 0, c, :] = scaler.transform(X_test[:, 0, c, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换为 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)


# =========================================================
# 3. SINGLE-BAND EEG NET MODEL (单频带脑电时空卷积网络)
# =========================================================
class SingleBandEEGNet(nn.Module):
    def __init__(self, num_classes, num_channels=60):
        super().__init__()

        # 输入形状: (B, 1, Channels, Time)

        # 1. 时间卷积：在时间轴上提取波形特征（替代传统滤波，让网络自己学滤波器）
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 2. 空间卷积：融合通道信息（把通道维压缩掉）
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # 3. 经典的 Variance Pooling层
        # 经过空间卷积后，形状变为 (B, 32, 1, T)

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.temporal(x)  # (B, 16, Channels, T)
        h = self.spatial(h)  # (B, 32, 1, T)

        # Variance Pooling: 沿时间轴(dim=-1)计算方差并降维
        h = torch.var(h, dim=-1).squeeze(-1)  # (B, 32)
        h = torch.log(torch.clamp(h, min=1e-6))

        return self.classifier(h)


# =========================================================
# 4. TRAINING
# =========================================================
model = SingleBandEEGNet(num_classes=num_classes, num_channels=C).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

epochs = 100
print("\n🚀 开始训练（已去掉频带提取的单频带时空网络）...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    if epoch % 10 == 0 or epoch == epochs - 1:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        print(f"Epoch {epoch:3d} | Loss: {total_loss / len(X_train_t):.4f} | Test Acc: {correct / total:.4f}")

# =========================================================
# 5. FINAL EVALUATION & EER
# =========================================================
model.eval()
all_probs = []
all_preds = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)

final_probs = np.concatenate(all_probs, axis=0)
final_preds = np.concatenate(all_preds, axis=0)
y_true = y_test

acc = accuracy_score(y_true, final_preds)
rec = recall_score(y_true, final_preds, average='macro')
f1 = f1_score(y_true, final_preds, average='macro')

# EER 计算
y_onehot = np.eye(num_classes)[y_true]
fpr, tpr, _ = roc_curve(y_onehot.ravel(), final_probs.ravel())
eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)

print("\n========== 最终评估结果 ==========")
print(f"ACC: {acc:.4f}")
print(f"REC: {rec:.4f}")
print(f"F1 : {f1:.4f}")
print(f"EER: {eer:.4f}")
print("===================================")