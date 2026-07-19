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
from vit_pytorch import ViT

# =========================================================
# 1. LOAD DATA (直接从原始 MAT 矩阵加载时域信号)
# =========================================================
mat_path = "../../../阅读阶段/EEG_deaf.mat"
var_name = "EEG_deaf"

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
# 确保每个被试的 64 个 Trial 里，都有 30% 进入测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================================
# 3. NORMALIZATION (时域 Z-score 标准化)
# =========================================================
N_train = X_train.shape[0]
N_test = X_test.shape[0]

scaler = StandardScaler()
# 展平后进行全局标准化，再恢复为 (60, 1600) 的 2D 时域图
X_train_scaled = scaler.fit_transform(X_train.reshape(N_train, -1)).reshape(N_train, C, T)
X_test_scaled = scaler.transform(X_test.reshape(N_test, -1)).reshape(N_test, C, T)

# =========================================================
# 4. ADD CHANNEL DIM & CONVERT TO TENSOR
# =========================================================
# 增加一个单通道维度符合 ViT 灰度图输入格式 -> (N, 1, 60, 1600)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# =========================================================
# 5. FIXED VIT (针对 60×1600 原始时域图动态调优)
# =========================================================
class EEGViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = ViT(
            image_size=(60, 1600),   # 对应原始时域图尺寸 (Channels, TimePoints)
            patch_size=(10, 80),     # 完美整除 (60, 1600) -> 60/10=6, 1600/80=20 块
            num_classes=num_classes, # 22 类
            channels=1,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.2,
            emb_dropout=0.2
        )

    def forward(self, x):
        return self.vit(x)

# =========================================================
# 6. TRAIN
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGViT(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# 送入设备
X_train_tensor = X_train_tensor.to(device)
X_test_tensor  = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

batch_size = 32
epochs = 120

print("\n🚀 开始训练原始时域脑电 ViT 模型...")
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(X_train_tensor))
    total_loss = 0

    for i in range(0, len(X_train_tensor), batch_size):
        idx = perm[i:i+batch_size]
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
        print(f"Epoch {epoch+1:03d} | Loss {total_loss:.3f} | Test Acc {acc:.4f}")

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
final_f1  = f1_score(y_true, preds, average='macro')

# 2. 计算封闭集 1-vs-All 统一等错误率 (EER)
y_true_onehot = np.eye(num_classes)[y_true]
y_true_flat = y_true_onehot.ravel()
y_pred_flat = probs.ravel()

fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print("\n" + "="*30)
print("       TEST SET METRICS")
print("="*30)
print(f"Accuracy  (ACC) : {final_acc:.4f}")
print(f"Recall    (REC) : {final_rec:.4f}")
print(f"F1-Score  (F1)  : {final_f1:.4f}")
print(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
print("="*30)

# =========================================================
# 8. 模型保存
# =========================================================
model_path = "mem_raw_eeg_vit_deaf.pth"
torch.save(model.state_dict(), model_path)
print(f"✔ Model weights successfully saved to: '{model_path}'")
print("="*30)