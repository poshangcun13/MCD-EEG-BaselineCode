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
from vit_pytorch import ViT

# =========================================================
# 1. LOAD DATA (跨任务协议: 记忆阶段 -> 阅读阶段)
# =========================================================
test_mat_path = "../记忆阶段/EEG_deaf_memory.mat"
train_mat_path = "../IQ/EEG_deaf_IQ.mat"

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

print(f"Train (Memory) Samples: {X_train_raw.shape[0]}")
print(f"Test  (Reading) Samples: {X_test_raw.shape[0]}")
print(f"Channels: {C_tr}, TimePoints: {T_tr}, Subjects: {num_classes}")

# =========================================================
# 2. CROSS-TASK NORMALIZATION (严格跨任务对齐)
# =========================================================
N_train = X_train_raw.shape[0]
N_test = X_test_raw.shape[0]

# scaler = StandardScaler()
# # 在记忆阶段数据上 Fit 均值和方差，并应用到双方
# X_train_scaled = scaler.fit_transform(X_train_raw.reshape(N_train, -1)).reshape(N_train, C_tr, T_tr)
# X_test_scaled = scaler.transform(X_test_raw.reshape(N_test, -1)).reshape(N_test, C_te, T_te)

X_train_scaled = X_train_raw
X_test_scaled  = X_test_raw


# =========================================================
# 3. ADD CHANNEL DIM & CONVERT TO TENSOR
# =========================================================
# 转换为标准 4D Tensor -> (样本数, 1, 60, 1600)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test_raw, dtype=torch.long)

# =========================================================
# 4. FIXED VIT (针对 60×1600 原始时域图调优)
# =========================================================
class EEGViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = ViT(
            image_size=(60, 1600),   # 原始时域 2D 排布 (Channels, TimePoints)
            patch_size=(10, 80),     # 完美整除 (60, 1600)
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
# 5. TRAINING PIPELINE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGViT(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# 送入 GPU/CPU 设备
X_train_tensor = X_train_tensor.to(device)
X_test_tensor  = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

batch_size = 32
epochs = 100

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
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor).argmax(dim=1)
        acc = (pred == y_test_tensor).float().mean().item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
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
model_path = os.path.join(save_dir, "cross_task_mem_to_read_vit.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("="*45)