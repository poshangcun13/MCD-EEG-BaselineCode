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
# 保持单通道 4D 格式输入 -> (N, 1, Channels, TimePoints)
X_train_tensor = torch.tensor(X_train_scaled[:, None, :, :], dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled[:, None, :, :], dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# =========================================================
# 5. EEGPT MODEL IMPLEMENTATION (Decoder-Only Generative/Representation Architecture)
# =========================================================
class EEGPT(nn.Module):
    def __init__(self, num_classes, chans=60, times=1600, emb_size=64, depth=6, heads=8, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        # 1. Token Embedding Layer: 将脑电通过一维卷积切成时间块（Token化）
        # 输入形状: (Batch, chans, times)
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
        # 核心点在于训练/前向传播时，必须使用自回归的 Causal Attention 掩码
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=emb_size * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True,  # 格式为 (Batch, Seq, Feature)
            norm_first=True  # Pre-LN 结构更稳定，贴近标准 GPT
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
        x = self.patch_embed(x)  # 形状: (Batch, emb_size, seq_len)
        x = x.transpose(1, 2)  # 变形为标准 NLP 格式: (Batch, seq_len, emb_size)

        # 2. 加上位置编码
        x = x + self.pos_embed

        # 3. 构建 Causal Mask (生成式下三角掩码，避免看到未来的脑电数据)
        # 虽然是做分类，但 GPT 架构的灵魂就是严格的因果注意力机制
        device = x.device
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len, device=device)

        # 4. 送入 GPT 骨干网络
        for block in self.transformer_blocks:
            # 显式传入 is_causal=True 或 mask，将其激活为自回归模式
            x = block(x, src_mask=mask, is_causal=True)

        x = self.ln_f(x)  # 形状: (Batch, seq_len, emb_size)

        # 5. 池化/提取序列末尾或全局表征进行分类 (这里采用末尾/平均池化复合特征)
        # GPT 通常取最后一个 Token 的特征，或者用 Mean Pooling
        x_mean = x.mean(dim=1)  # 提取全局全局时序均值 (Batch, emb_size)

        out = self.classifier(x_mean)
        return out


# =========================================================
# 6. TRAIN
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化 EEGPT 模型
model = EEGPT(num_classes=num_classes, chans=C, times=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

# 送入设备
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

batch_size = 32
epochs = 200

print("\n🚀 开始训练原始时域脑电 EEGPT 模型...")
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
print("       TEST SET METRICS (EEGPT)")
print("=" * 30)
print(f"Accuracy  (ACC) : {final_acc:.4f}")
print(f"Recall    (REC) : {final_rec:.4f}")
print(f"F1-Score  (F1)  : {final_f1:.4f}")
print(f"Equal Error Rate (EER): {eer:.4f} ({eer * 100:.2f}%)")
print("=" * 30)

# =========================================================
# 8. 模型保存
# =========================================================
model_path = "mem_raw_eeg_gpt_deaf.pth"
torch.save(model.state_dict(), model_path)
print(f"✔ Model weights successfully saved to: '{model_path}'")
print("=" * 30)