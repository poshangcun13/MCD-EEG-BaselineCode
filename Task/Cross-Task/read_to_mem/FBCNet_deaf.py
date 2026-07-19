import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter

# =========================================================
# 1. LOAD DATA (跨任务协议: 记忆阶段 -> 阅读阶段)
# =========================================================
test_mat_path = "../../../记忆阶段/EEG_deaf_memory.mat"
train_mat_path = "../../../阅读阶段/EEG_deaf.mat"

print("=" * 70)
print(f"🚀 SOURCE TASK (Train) : {train_mat_path}")
print(f"🚀 TARGET TASK (Test)  : {test_mat_path}")
print("=" * 70)

# ---- 1.1 加载训练数据（记忆阶段） ----
mat_train = sio.loadmat(train_mat_path)
raw_train = mat_train["EEG_deaf"]  # 形状: (22, 60, 1600, 64)
S_tr, C_tr, T_tr, N_tr = raw_train.shape

# 维度重排与样本展平 -> 形状: (1408, 60, 1600)
X_train_perm = np.transpose(raw_train, (0, 3, 1, 2))
X_train_raw = X_train_perm.reshape(S_tr * N_tr, C_tr, T_tr)
y_train_raw = np.repeat(np.arange(S_tr), N_tr)

# ---- 1.2 加载测试数据（阅读阶段） ----
mat_test = sio.loadmat(test_mat_path)
raw_test = mat_test["EEG_deaf"]  # 形状: (22, 60, 1600, 64)
S_te, C_te, T_te, N_te = raw_test.shape

# 维度重排与样本展平 -> 形状: (1408, 60, 1600)
X_test_perm = np.transpose(raw_test, (0, 3, 1, 2))
X_test_raw = X_test_perm.reshape(S_te * N_te, C_te, T_te)
y_test_raw = np.repeat(np.arange(S_te), N_te)

# 安全校验
assert S_tr == S_te, "训练集与测试集的被试类别数量不一致！"
num_classes = S_tr
C = C_tr
T = T_tr

print(f"Train (Memory) Samples: {X_train_raw.shape[0]}")
print(f"Test  (Reading) Samples: {X_test_raw.shape[0]}")
print(f"Channels: {C}, TimePoints: {T}, Subjects: {num_classes}")

# =========================================================
# 2. MULTI-BAND FILTERING (多频带提取预处理)
# =========================================================
bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def butter_bandpass_filter(data, lowcut, highcut, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)


print("\n正在对源域（训练集）与目标域（测试集）分别进行多频带滤波...")
X_train_bands = []
X_test_bands = []

for low, high in bands:
    X_train_bands.append(butter_bandpass_filter(X_train_raw, low, high))
    X_test_bands.append(butter_bandpass_filter(X_test_raw, low, high))

# 堆叠频带维度，形状变为: (Samples, Bands, Channels, Time)
X_train_multi = np.stack(X_train_bands, axis=1).astype(np.float32)
X_test_multi = np.stack(X_test_bands, axis=1).astype(np.float32)

print(f"✔ 滤波完成！训练集形状: {X_train_multi.shape} | 测试集形状: {X_test_multi.shape}")

# =========================================================
# 3. CROSS-TASK Z-SCORE (严格跨任务对齐)
# =========================================================
# 针对每个通道进行标准化，在 Train 上 fit，在 Test 上 transform，防止未来数据泄露
for c in range(C):
    scaler = StandardScaler()
    # 展平频带和时间点，对该通道特征建立基准
    X_train_multi[:, :, c, :] = scaler.fit_transform(
        X_train_multi[:, :, c, :].reshape(-1, T)
    ).reshape(X_train_multi.shape[0], len(bands), T)

    X_test_multi[:, :, c, :] = scaler.transform(
        X_test_multi[:, :, c, :].reshape(-1, T)
    ).reshape(X_test_multi.shape[0], len(bands), T)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换成 Tensor
X_train_t = torch.tensor(X_train_multi, dtype=torch.float32)
y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
X_test_t = torch.tensor(X_test_multi, dtype=torch.float32)
y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)


# =========================================================
# 4. FBCNET MODEL (多频带脑电时空卷积网络)
# =========================================================
class FBCNet(nn.Module):
    def __init__(self, num_classes, num_bands=5, num_channels=60):
        super().__init__()

        # 1. 时间卷积 (使用 groups 实现独立频带并行处理)
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

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * num_bands, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.temporal(x)
        h = self.spatial(h)

        # Variance Pooling: 沿时间轴计算方差
        h = torch.var(h, dim=-1).squeeze(-1)
        h = torch.log(torch.clamp(h, min=1e-6))

        return self.classifier(h)


# =========================================================
# 5. TRAINING PIPELINE
# =========================================================
model = FBCNet(num_classes=num_classes, num_bands=len(bands), num_channels=C).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

epochs = 100  # 调整为 100 epoch，以便与整体流程对齐
print("\n🚀 开始执行跨任务脑纹识别训练 (Reading -> Memory)...")

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

    # 验证在全新目标域（阅读阶段）下的泛化表现
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
        print(
            f"Epoch {epoch:03d} | Train Loss: {total_loss / len(X_train_t):.4f} | Cross-Task Test Acc: {correct / total:.4f}")

# =========================================================
# 6. FINAL EVALUATION & EER (跨任务泛化能力最终测试)
# =========================================================
print("\n正在执行目标域 (Reading Task) 最终指标评估...")
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
y_true = y_test_raw

# 1. 计算核心多分类泛化指标
final_acc = accuracy_score(y_true, final_preds)
final_rec = recall_score(y_true, final_preds, average='macro')
final_f1 = f1_score(y_true, final_preds, average='macro')

# 2. 计算跨场景身份验证的等错误率 (EER)
y_onehot = np.eye(num_classes)[y_true]
fpr, tpr, _ = roc_curve(y_onehot.ravel(), final_probs.ravel())
eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)

print("\n" + "=" * 45)
print("       CROSS-TASK (MEM -> READ) METRICS REPORT")
print("=" * 45)
print(f"Cross-Task Accuracy  (ACC) : {final_acc:.4f}")
print(f"Cross-Task Recall    (REC) : {final_rec:.4f}")
print(f"Cross-Task F1-Score  (F1)  : {final_f1:.4f}")
print(f"Cross-Task EER       (EER) : {eer:.4f} ({eer * 100:.2f}%)")
print("===================================")

# =========================================================
# 7. 模型保存
# =========================================================
save_dir = "./"
model_path = os.path.join(save_dir, "cross_task_mem_to_read_fbcnet.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("=" * 45)