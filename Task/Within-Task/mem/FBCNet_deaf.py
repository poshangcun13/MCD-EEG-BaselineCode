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
from scipy.signal import butter, lfilter

# =========================================================
# 1. LOAD & FILTER DATA
# =========================================================
mat_path = "../../../记忆阶段/EEG_deaf_memory.mat"
var_name = "EEG_deaf"

mat_data = sio.loadmat(mat_path)
raw_matrix = mat_data[var_name]  # (22, 60, 1600, 64)

S, C, T, N = raw_matrix.shape
num_classes = S

X = np.transpose(raw_matrix, (0, 3, 1, 2)).reshape(S * N, C, T)
y = np.repeat(np.arange(S), N)

bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def butter_bandpass_filter(data, lowcut, highcut, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)


print("正在预处理滤波（多频带提取）...")
X_bands = []
for low, high in bands:
    filtered = butter_bandpass_filter(X, low, high)
    X_bands.append(filtered)

X_multi_band = np.stack(X_bands, axis=1)
print("预处理完成，数据形状:", X_multi_band.shape)

# =========================================================
# 2. SPLIT & Z-SCORE
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_multi_band, y, test_size=0.3, random_state=42, stratify=y
)

for c in range(C):
    scaler = StandardScaler()
    X_train[:, :, c, :] = scaler.fit_transform(X_train[:, :, c, :].reshape(-1, T)).reshape(X_train.shape[0], len(bands),
                                                                                           T)
    X_test[:, :, c, :] = scaler.transform(X_test[:, :, c, :].reshape(-1, T)).reshape(X_test.shape[0], len(bands), T)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换成 Tensor 暂存到 CPU 内存中（防止直接塞满显存）
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 训练集和测试集都用 DataLoader 进行管理（测试集不 shuffle）
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)


# =========================================================
# 3. FBCNET MODEL
# =========================================================
class FBCNet(nn.Module):
    def __init__(self, num_classes, num_bands=5, num_channels=60):
        super().__init__()

        # 1. 时间卷积
        self.temporal = nn.Sequential(
            nn.Conv2d(num_bands, 16 * num_bands, kernel_size=(1, 64), padding=(0, 32), groups=num_bands),
            nn.BatchNorm2d(16 * num_bands),
            nn.ELU()
        )

        # 2. 空间卷积
        self.spatial = nn.Sequential(
            nn.Conv2d(16 * num_bands, 32 * num_bands, kernel_size=(num_channels, 1), groups=num_bands),
            nn.BatchNorm2d(32 * num_bands),
            nn.ELU()
        )

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * num_bands, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.temporal(x)
        h = self.spatial(h)

        # Variance Pooling
        h = torch.var(h, dim=-1).squeeze(-1)
        h = torch.log(torch.clamp(h, min=1e-6))

        return self.classifier(h)


# =========================================================
# 4. TRAINING WITH BATCH EVALUATION
# =========================================================
model = FBCNet(num_classes=num_classes, num_bands=len(bands), num_channels=C).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

epochs = 20
print("\n🚀 开始训练优化后的 FBCNet...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    # 训练循环
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)  # 在这里将小 batch 移到 GPU
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    # 每 10 轮用批处理方式评估一次测试集，绝不溢出
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
# 5. FINAL EVALUATION & EER CALCULATING
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

# 拼接所有 batch 的结果
final_probs = np.concatenate(all_probs, axis=0)
final_preds = np.concatenate(all_preds, axis=0)
y_true = y_test

# 指标计算
acc = accuracy_score(y_true, final_preds)
rec = recall_score(y_true, final_preds, average='macro')
f1 = f1_score(y_true, final_preds, average='macro')

# 针对脑纹识别（身份鉴别）的多分类 EER 计算
y_onehot = np.eye(num_classes)[y_true]
fpr, tpr, _ = roc_curve(y_onehot.ravel(), final_probs.ravel())
eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)

print("\n========== 最终评估结果 ==========")
print(f"ACC: {acc:.4f}")
print(f"REC: {rec:.4f}")
print(f"F1 : {f1:.4f}")
print(f"EER: {eer:.4f}")
print("===================================")