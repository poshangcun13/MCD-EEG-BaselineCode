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

# 维度重排与样本展平 -> 形状: (1408, 1, 60, 1600)
X_train_perm = np.transpose(raw_train, (0, 3, 1, 2))
X_train_raw = X_train_perm.reshape(S_tr * N_tr, 1, C_tr, T_tr).astype(np.float32)
y_train_raw = np.repeat(np.arange(S_tr), N_tr)

# ---- 1.2 加载测试数据（阅读阶段） ----
mat_test = sio.loadmat(test_mat_path)
raw_test = mat_test["EEG_deaf"]  # 形状: (22, 60, 1600, 64)
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
# 针对原始信号进行通道内归一化。严格在 Train 上 fit，在 Test 上 transform
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
# 3. ATCNET MODEL IMPLEMENTATION (Attention TCN 架构)
# =========================================================
class TCNBlock(nn.Module):
    """ 因果扩张时域卷积块 (Causal Dilated Conv Block) """
    def __init__(self, in_channels, out_channels, kernel_size=4, dilation=2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.padding = padding

    def forward(self, x):
        res = self.residual(x)
        out = self.conv(x)
        # 截断右侧多余的 Padding，确保 Causal 因果机制
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return self.elu(self.bn(out) + res)


class ATCNet(nn.Module):
    def __init__(self, num_classes, num_channels=60, eeg_time_points=1600):
        super().__init__()

        # 1. 浅层时空 Conv 模块
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.elu = nn.ELU()

        # 降低时间分辨率 (1600 // 8 = 200)
        self.pool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        reduced_time = eeg_time_points // 8

        # 2. 多头自注意力模块 (Multi-Head Self-Attention)
        self.mhsa = nn.MultiheadAttention(embed_dim=32, num_heads=4, dropout=0.3, batch_first=True)
        self.norm_sa = nn.LayerNorm(32)

        # 3. 时域卷积网络模块 (TCN Block)
        self.tcn1 = TCNBlock(in_channels=32, out_channels=32, kernel_size=4, dilation=1)
        self.tcn2 = TCNBlock(in_channels=32, out_channels=64, kernel_size=4, dilation=2)

        # 4. 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * reduced_time, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. 浅层时空图提取
        h = self.temporal_conv(x)            # (B, 16, Channels, T)
        h = self.spatial_conv(h)             # (B, 32, 1, T)
        h = self.pool(self.elu(self.bn(h)))  # (B, 32, 1, T_reduced)
        h = h.squeeze(2)                     # (B, 32, T_reduced)

        # 2. 转换形状以输入 Attention 模块
        h_sa = h.permute(0, 2, 1)            # (B, T_reduced, 32)
        attn_out, _ = self.mhsa(h_sa, h_sa, h_sa)
        h_sa = self.norm_sa(attn_out + h_sa) # 残差连接与归一化

        # 3. 转换回一维卷积形状输入 TCN
        h_tcn = h_sa.permute(0, 2, 1)        # (B, 32, T_reduced)
        h_tcn = self.tcn1(h_tcn)             # (B, 32, T_reduced)
        h_tcn = self.tcn2(h_tcn)             # (B, 64, T_reduced)

        # 4. 展平分类
        h_flat = h_tcn.view(h_tcn.size(0), -1)
        return self.classifier(h_flat)


# =========================================================
# 4. TRAINING WITH MIXED PRECISION (AMP 混合精度跨任务训练)
# =========================================================
model = ATCNet(num_classes=num_classes, num_channels=C, eeg_time_points=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 启用自动混合精度缩放
scaler = torch.cuda.amp.GradScaler()

epochs = 100  # 调整为 100 epoch 保持实验基准一致
print("\n🚀 开始执行跨任务脑纹识别训练 (Reading -> Memory)...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

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
model_path = os.path.join(save_dir, "cross_task_mem_to_read_atcnet.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("=" * 45)