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
torch.backends.cudnn.benchmark = True  # 开启算子硬件动态优化

# 转换为 Tensor 并在显存内连续化存放（消除零散拷贝造成的延迟）
X_train_t = torch.tensor(X_train, dtype=torch.float32).contiguous()
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).contiguous()
y_test_t = torch.tensor(y_test, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)


# =========================================================
# 3. LOW-RANK MATRIX DECOMPOSITION MODEL (LRMD 脑电解耦网络)
# =========================================================
class LRMD_EEGNet(nn.Module):
    def __init__(self, num_classes, num_channels=60, time_points=1600, rank=16):
        """
        rank: 低秩约束矩阵的秩 (Rank)。值越小，解耦过滤任务扰动的能力越强，且运算速度越快。
        """
        super().__init__()

        # 1. 空间轻量化压缩层：快速跨通道整合，不使用时间轴滑窗，彻底根除时域卷积瓶颈
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 展平后的维度: 16特征图 * 1 * 1600时间点 = 25600
        self.flatten_dim = 16 * 1 * time_points
        self.hidden_dim = 128

        # 2. 【LRMD 低秩分解核心模块】
        # 将一个大映射矩阵 W (flatten_dim -> hidden_dim) 拆解为两个超轻量低秩矩阵的乘积：
        # W = low_rank_A * low_rank_B
        # 特征流先被强行压缩到维度为 rank 的窄空间，剔除稀疏任务扰动，提取固有低秩背景脑纹骨架
        self.low_rank_A = nn.Linear(self.flatten_dim, rank, bias=False)
        self.low_rank_B = nn.Linear(rank, self.hidden_dim, bias=True)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.4)

        # 3. 分类器
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # 输入 x 形状: (B, 1, Channels, Time)
        h = self.spatial_compress(x)  # 形状变换: (B, 16, 1, T)
        h = h.view(h.size(0), -1)  # 展平: (B, 16 * 1 * T)

        # 低秩分解特征变换
        h_rank = self.low_rank_A(h)  # 投影到低秩瓶颈空间: (B, rank)
        h_lrmd = self.low_rank_B(h_rank)  # 重新放大回隐藏层空间: (B, hidden_dim)
        h_lrmd = self.dropout(self.elu(h_lrmd))

        return self.classifier(h_lrmd)


# =========================================================
# 4. TRAINING WITH AMP
# =========================================================
model = LRMD_EEGNet(num_classes=num_classes, num_channels=C, time_points=T, rank=16).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 初始化 AMP 自动混合精度梯度缩放器，让网络乘法运算自动使用 Float16 计算，速度狂飙
scaler = torch.cuda.amp.GradScaler()

epochs = 200
print("\n🚀 开始训练（基于混合精度与 LRMD 低秩矩阵自适应网络）...")

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

print("\n========== 最终评估结果 (LRMD Architecture) ==========")
print(f"ACC: {acc:.4f}")
print(f"REC: {rec:.4f}")
print(f"F1 : {f1:.4f}")
print(f"EER: {eer:.4f}")
print("======================================================")