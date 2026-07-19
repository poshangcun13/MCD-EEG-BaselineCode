import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

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

# 维度重排与样本展平 -> 形状: (1408, 60, 1600)
X_train_perm = np.transpose(raw_train, (0, 3, 1, 2))
X_train_raw = X_train_perm.reshape(S_tr * N_tr, C_tr, T_tr)
y_train_raw = np.repeat(np.arange(S_tr), N_tr)

# ---- 1.2 加载测试数据（阅读阶段） ----
mat_test = sio.loadmat(test_mat_path)
raw_test = mat_test["EEG_hearing"]  # 形状: (22, 60, 1600, 64)
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
# 2. CROSS-TASK NORMALIZATION (严格跨任务对齐)
# =========================================================
N_train = X_train_raw.shape[0]
N_test = X_test_raw.shape[0]

scaler = StandardScaler()
# 在记忆阶段数据上 Fit 均值和方差，并应用到双方，杜绝数据泄露
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(N_train, -1)).reshape(N_train, C, T)
X_test_scaled = scaler.transform(X_test_raw.reshape(N_test, -1)).reshape(N_test, C, T)

# =========================================================
# 3. ADD CHANNEL DIM & CONVERT TO TENSOR
# =========================================================
# 符合 EEGConformer 标准输入格式 -> (样本数, 1, Channels, TimePoints)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test_raw, dtype=torch.long)

# =========================================================
# 4. EEGConformer MODEL IMPLEMENTATION
# =========================================================
class EEGConformer(nn.Module):
    def __init__(self, num_classes, chans=60, times=1600, emb_size=40, depth=6, heads=8):
        super().__init__()

        # 1. 浅层一维时间卷积 (Temporal Component)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(emb_size),
            nn.ELU()
        )

        # 2. 空间深度卷积 (Spatial Component)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(chans, 1), stride=(1, 1), groups=emb_size),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),  # 降低时间分辨率
            nn.Dropout(0.2)
        )

        # 计算经过卷积池化后的时间步长 (1600 / 8 = 200)
        reduced_times = times // 8

        # 3. Transformer 模块 (Transformer Component)
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
        x = self.spatial_conv(x)  # 形状: [Batch, emb_size, 1, reduced_times]

        # 2. 维度调整以符合 Transformer 的输入要求
        x = x.squeeze(2)          # 形状: [Batch, emb_size, reduced_times]
        x = x.permute(2, 0, 1)    # 形状: [reduced_times, Batch, emb_size]

        # 3. Transformer 编码
        x = self.transformer(x)   # 形状: [reduced_times, Batch, emb_size]

        # 4. 展平并分类
        x = x.permute(1, 0, 2)    # 形状: [Batch, reduced_times, emb_size]
        x = x.reshape(x.size(0), -1)  # 展平为二维

        out = self.classifier(x)
        return out

# =========================================================
# 5. TRAINING PIPELINE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型并传给设备
model = EEGConformer(num_classes=num_classes, chans=C, times=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 送入显存/内存
X_train_tensor = X_train_tensor.to(device)
X_test_tensor  = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

batch_size = 32
epochs = 200

print("\n🚀 开始执行跨任务脑纹识别训练 (Memory -> Recap)...")
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

    # 验证在全新任务（阅读阶段）下的泛化识别性能
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).argmax(dim=1)
            acc = (pred == y_test_tensor).float().mean().item()
        print(f"Epoch {epoch+1:03d} | Train Loss {total_loss:.3f} | Cross-Task Test Acc {acc:.4f}")

# =========================================================
# 6. FINAL EVALUATION (跨任务泛化能力测试)
# =========================================================
print("\n正在执行目标域 (Reading Task) 最终指标评估...")
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

y_true = y_test_tensor.cpu().numpy()

# 1. 计算核心多分类泛化指标
final_acc = accuracy_score(y_true, preds)
final_rec = recall_score(y_true, preds, average='macro')
final_f1  = f1_score(y_true, preds, average='macro')

# 2. 计算跨场景 1-vs-All 身份验证等错误率 (EER)
y_true_onehot = np.eye(num_classes)[y_true]
y_true_flat = y_true_onehot.ravel()
y_pred_flat = probs.ravel()

fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print("\n" + "="*45)
print("       CROSS-TASK (MEM -> READ) METRICS REPORT")
print("="*45)
print(f"Cross-Task Accuracy  (ACC) : {final_acc:.4f}")
print(f"Cross-Task Recall    (REC) : {final_rec:.4f}")
print(f"Cross-Task F1-Score  (F1)  : {final_f1:.4f}")
print(f"Cross-Task EER       (EER) : {eer:.4f} ({eer*100:.2f}%)")
print("="*45)

# =========================================================
# 7. 模型保存
# =========================================================
save_dir = "./"
model_path = os.path.join(save_dir, "cross_task_mem_to_read_conformer.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("="*45)