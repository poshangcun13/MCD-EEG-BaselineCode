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
# 在记忆阶段数据上 Fit 均值和方差，并应用到双方，保证跨任务测试的严谨性
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(N_train, -1)).reshape(N_train, C, T)
X_test_scaled = scaler.transform(X_test_raw.reshape(N_test, -1)).reshape(N_test, C, T)

# =========================================================
# 3. ADD CHANNEL DIM & CONVERT TO TENSOR
# =========================================================
# 保持单通道 4D 格式输入 -> (样本数, 1, Channels, TimePoints)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test_raw, dtype=torch.long)

# =========================================================
# 4. EEGPT MODEL IMPLEMENTATION
# =========================================================
class EEGPT(nn.Module):
    def __init__(self, num_classes, chans=60, times=1600, emb_size=64, depth=6, heads=8, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        # 1. Token Embedding Layer: 将脑电通过一维卷积切成时间块（Token化）
        self.patch_embed = nn.Conv1d(
            in_channels=chans,
            out_channels=emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 计算 Token 序列的总长度 (1600 / 8 = 200)
        self.seq_len = times // patch_size

        # 可学习的位置编码 (Position Embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, emb_size))

        # 2. GPT Decoder Layers (Causal Transformer Block)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=emb_size * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 结构更稳定
        )
        self.transformer_blocks = nn.ModuleList([decoder_layer for _ in range(depth)])
        self.ln_f = nn.LayerNorm(emb_size)

        # 3. 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 移除无用的 1 维 -> (Batch, Chans, Times)
        x = x.squeeze(1)

        # 1. 脑电局部时空 Token 提取
        x = self.patch_embed(x)    # 形状: (Batch, emb_size, seq_len)
        x = x.transpose(1, 2)      # 变形为 NLP 格式: (Batch, seq_len, emb_size)

        # 2. 加上位置编码
        x = x + self.pos_embed

        # 3. 构建 Causal Mask (下三角掩码，避免看到未来的脑电数据)
        device = x.device
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len, device=device)

        # 4. 送入 GPT 骨干网络
        for block in self.transformer_blocks:
            x = block(x, src_mask=mask, is_causal=True)

        x = self.ln_f(x)  # 形状: (Batch, seq_len, emb_size)

        # 5. 池化特征用于多分类
        x_mean = x.mean(dim=1)  # 提取全局时序均值 (Batch, emb_size)

        out = self.classifier(x_mean)
        return out

# =========================================================
# 5. TRAINING PIPELINE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化 EEGPT 模型
model = EEGPT(num_classes=num_classes, chans=C, times=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

# 送入设备
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

    # 验证在全新任务（阅读阶段）下的跨域识别性能
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
model_path = os.path.join(save_dir, "cross_task_mem_to_read_eeggpt.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("="*45)