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

# =========================================================
# 1. LOAD DATA (跨任务协议: 记忆阶段 -> 阅读阶段)
# =========================================================
test_mat_path = "../记忆阶段/EEG_hearing_memory.mat"
train_mat_path = "../IQ/EEG_hearing_IQ.mat"

print("=" * 70)
print(f"🚀 SOURCE TASK (Train) : {train_mat_path}")
print(f"🚀 TARGET TASK (Test)  : {test_mat_path}")
print("=" * 70)

# ---- 1.1 加载训练数据（记忆阶段） ----
mat_train = sio.loadmat(train_mat_path)
raw_train = mat_train["EEG_hearing"]  # 形状: (22, 60, 1600, 64)
S_tr, C_tr, T_tr, N_tr = raw_train.shape

# 维度重排与样本展平 -> 形状: (1408, 1, 60, 1600)
X_train_perm = np.transpose(raw_train, (0, 3, 1, 2))
X_train_raw = X_train_perm.reshape(S_tr * N_tr, 1, C_tr, T_tr).astype(np.float32)
y_train_raw = np.repeat(np.arange(S_tr), N_tr)

# ---- 1.2 加载测试数据（阅读阶段） ----
mat_test = sio.loadmat(test_mat_path)
raw_test = mat_test["EEG_hearing"]  # 形状: (22, 60, 1600, 64)
S_te, C_te, T_te, N_te = raw_test.shape

# 维度重排与样本展平 -> 形状: (1408, 1, 60, 1600)
X_test_perm = np.transpose(raw_test, (0, 3, 1, 2))
X_test_raw = X_test_perm.reshape(S_te * N_te, 1, C_te, T_te).astype(np.float32)
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
# 2. CROSS-TASK CHANNEL-WISE NORMALIZATION (严格跨任务通道对齐)
# =========================================================
# 针对原始信号，进行通道内归一化。严格在 Train 上 fit，在 Test 上 transform
for c in range(C):
    scaler = StandardScaler()
    # 提取第 c 个通道的数据，形状为 (Samples, Time)
    X_train_raw[:, 0, c, :] = scaler.fit_transform(X_train_raw[:, 0, c, :])
    X_test_raw[:, 0, c, :] = scaler.transform(X_test_raw[:, 0, c, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 开启算子硬件动态优化

# 转换为 Tensor 并在显存内连续化存放
X_train_t = torch.tensor(X_train_raw, dtype=torch.float32).contiguous()
y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
X_test_t = torch.tensor(X_test_raw, dtype=torch.float32).contiguous()
y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)


# =========================================================
# 3. LOW-RANK MATRIX DECOMPOSITION MODEL (LRMD 脑电解耦网络)
# =========================================================
class LRMD_EEGNet(nn.Module):
    def __init__(self, num_classes, num_channels=60, time_points=1600, rank=16):
        super().__init__()

        # 1. 空间轻量化压缩层
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        self.flatten_dim = 16 * 1 * time_points
        self.hidden_dim = 128

        # 2. LRMD 低秩分解核心模块
        self.low_rank_A = nn.Linear(self.flatten_dim, rank, bias=False)
        self.low_rank_B = nn.Linear(rank, self.hidden_dim, bias=True)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.4)

        # 3. 分类器
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        h = self.spatial_compress(x)  # (B, 16, 1, T)
        h = h.view(h.size(0), -1)      # 展平: (B, 16 * 1 * T)

        # 低秩分解特征变换
        h_rank = self.low_rank_A(h)    # 投影到低秩瓶颈空间: (B, rank)
        h_lrmd = self.low_rank_B(h_rank)  # 重新放大回隐藏层空间: (B, hidden_dim)
        h_lrmd = self.dropout(self.elu(h_lrmd))

        return self.classifier(h_lrmd)


# =========================================================
# 4. TRAINING WITH AMP (混合精度跨任务训练)
# =========================================================
model = LRMD_EEGNet(num_classes=num_classes, num_channels=C, time_points=T, rank=16).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 初始化 AMP 自动混合精度梯度缩放器
scaler = torch.cuda.amp.GradScaler()

epochs = 100  # 调整为 100 epoch 保持基准一致
print("\n🚀 开始执行跨任务脑纹识别训练 (Memory -> Recap)...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # 激活混合精度前向传播
        with torch.cuda.amp.autocast():
            out = model(xb)
            loss = criterion(out, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(xb)

    # 验证在全新任务（阅读阶段）下的跨域识别性能
    if epoch % 10 == 0 or epoch == epochs - 1:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast():
                    out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(X_train_t):.4f} | Cross-Task Test Acc: {correct / total:.4f}")

# =========================================================
# 5. FINAL EVALUATION & EER (跨任务泛化能力最终测试)
# =========================================================
print("\n正在执行目标域 (Reading Task) 最终指标评估...")
model.eval()
all_probs = []
all_preds = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        with torch.cuda.amp.autocast():
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
final_f1  = f1_score(y_true, final_preds, average='macro')

# 2. 计算跨场景身份验证的等错误率 (EER)
y_onehot = np.eye(num_classes)[y_true]
fpr, tpr, _ = roc_curve(y_onehot.ravel(), final_probs.ravel())
eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)

print("\n" + "="*45)
print("       CROSS-TASK (MEM -> READ) METRICS REPORT")
print("="*45)
print(f"Cross-Task Accuracy  (ACC) : {final_acc:.4f}")
print(f"Cross-Task Recall    (REC) : {final_rec:.4f}")
print(f"Cross-Task F1-Score  (F1)  : {final_f1:.4f}")
print(f"Cross-Task EER       (EER) : {eer:.4f} ({eer*100:.2f}%)")
print("===================================")

# =========================================================
# 6. 模型保存
# =========================================================
save_dir = "./"
model_path = os.path.join(save_dir, "cross_task_mem_to_read_lrmd.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("=" * 45)