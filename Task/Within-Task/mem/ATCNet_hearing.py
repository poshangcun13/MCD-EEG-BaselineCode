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
mat_path = "../../../记忆阶段/EEG_hearing_memory.mat"
var_name = "EEG_hearing"

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
    X_train[:, 0, c, :] = scaler.fit_transform(X_train[:, 0, c, :])
    X_test[:, 0, c, :] = scaler.transform(X_test[:, 0, c, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 转换为 Tensor 并在显存连续化
X_train_t = torch.tensor(X_train, dtype=torch.float32).contiguous()
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).contiguous()
y_test_t = torch.tensor(y_test, dtype=torch.long)

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
        # 为了保证因果性，非对称填充
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
        # 截断右侧多余的 Padding，确保 Causal 因果机制（不泄露未来信息）
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

        # 通过较大的平均池化平滑时域并下采样，降低多头注意力负担 (1600 // 8 = 200)
        self.pool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        reduced_time = eeg_time_points // 8

        # 2. 多头自注意力模块 (Multi-Head Self-Attention)
        # 期望形状: (Batch, Sequence_Length, Embedding_Dim) -> (B, reduced_time, 32)
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
        h = self.temporal_conv(x)  # (B, 16, Channels, T)
        h = self.spatial_conv(h)  # (B, 32, 1, T)
        h = self.pool(self.elu(self.bn(h)))  # (B, 32, 1, T_reduced)
        h = h.squeeze(2)  # (B, 32, T_reduced)

        # 2. 转换形状以输入 Attention 模块 (Batch, Seq_Len, Dim)
        h_sa = h.permute(0, 2, 1)  # (B, T_reduced, 32)
        attn_out, _ = self.mhsa(h_sa, h_sa, h_sa)
        h_sa = self.norm_sa(attn_out + h_sa)  # 残差连接与归一化

        # 3. 转换回一维卷积形状输入 TCN (Batch, Dim, Seq_Len)
        h_tcn = h_sa.permute(0, 2, 1)  # (B, 32, T_reduced)
        h_tcn = self.tcn1(h_tcn)  # (B, 32, T_reduced)
        h_tcn = self.tcn2(h_tcn)  # (B, 64, T_reduced)

        # 4. 展平分类
        h_flat = h_tcn.view(h_tcn.size(0), -1)
        return self.classifier(h_flat)


# =========================================================
# 4. TRAINING WITH MIXED PRECISION (AMP)
# =========================================================
model = ATCNet(num_classes=num_classes, num_channels=C, eeg_time_points=T).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 启用自动混合精度缩放，极大提升 ATCNet 多头注意力的显存运算速度
scaler = torch.cuda.amp.GradScaler()

epochs = 200
print("\n🚀 开始训练（单频带高效率 ATCNet 模型架构）...")

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
        with torch.cuda.amp.autocast():
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

print("\n========== 最终评估结果 (ATCNet) ==========")
print(f"ACC: {acc:.4f}")
print(f"REC: {rec:.4f}")
print(f"F1 : {f1:.4f}")
print(f"EER: {eer:.4f}")
print("===========================================")