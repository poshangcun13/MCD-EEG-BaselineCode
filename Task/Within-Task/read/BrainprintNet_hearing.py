import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import butter, lfilter, resample
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# =========================================================
# 1. 滤波器组与时域降采样预处理
# =========================================================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y


def create_filter_bank_with_downsample(X, fs=250, downsample_factor=2):
    """
    【物理级提速核心】：在构建FilterBank的同时进行时域降采样，极大缩减后续卷积的计算量
    """
    N_samples, C, T = X.shape
    bands = [(8 + 2 * i, 10 + 2 * i) for i in range(12)]
    N_bands = len(bands)

    # 计算降采样后的时间点数 (例如 1600 // 2 = 800)
    T_new = T // downsample_factor

    X_fb = np.zeros((N_samples, N_bands, C, T_new), dtype=np.float32)
    print(f"正在构建精细脑电滤波器组并降采样 (时间点: {T} -> {T_new})...")

    for b_idx, (low, high) in enumerate(bands):
        # 先进行带通滤波
        filtered = butter_bandpass_filter(X, low, high, fs, order=5)
        # 跨步切片进行极速降采样，规避重采样带来的额外 CPU 耗时
        X_fb[:, b_idx, :, :] = filtered[:, :, ::downsample_factor][:, :, :T_new]

    return X_fb, T_new


# =========================================================
# 2. LOAD & PREPROCESS DATA
# =========================================================
mat_path = "../../../阅读阶段/EEG_hearing.mat"
var_name = "EEG_hearing"

print("=" * 60)
print(f"Loading raw EEG MAT file from: {mat_path}")
print("=" * 60)

mat_data = sio.loadmat(mat_path)
raw_matrix = mat_data[var_name]  # (22, 60, 1600, 64)

S, C, T, N = raw_matrix.shape
num_classes = S

X_permuted = np.transpose(raw_matrix, (0, 3, 1, 2))
X_raw = X_permuted.reshape(S * N, C, T)
y = np.repeat(np.arange(S), N)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42, stratify=y
)

# 【提速开关】：设置时域降采样倍数。2表示时间点减半（1600->800），4表示变为四分之一（1600->400）
# 脑电识别在 400~800 点时信息已经高度冗余，降采样能极大加速训练
DOWNSAMPLE_FACTOR = 2
fs_eeg = 250

X_train_fb, T_net = create_filter_bank_with_downsample(X_train_raw, fs=fs_eeg, downsample_factor=DOWNSAMPLE_FACTOR)
X_test_fb, _ = create_filter_bank_with_downsample(X_test_raw, fs=fs_eeg, downsample_factor=DOWNSAMPLE_FACTOR)

X_train_tensor = torch.tensor(X_train_fb, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_fb, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"FilterBank 构建与降采样完毕。输入网络数据形状: {X_train_tensor.shape}")


# =========================================================
# 3. HIGH-PERFORMANCE BRAINPRINTNET IMPLEMENTATION
# =========================================================
class BrainprintNet(nn.Module):
    def __init__(self, num_classes, chans=60, n_bands=12, m=4, w=25):
        super().__init__()
        self.n_bands = n_bands
        self.chans = chans
        self.w = w  # 既然时间轴缩短了，Log-Variance 的滑窗大小也相应缩短（如 50 -> 25）
        self.Z = 3
        self.out_channels_per_scale = 8
        self.m = m
        self.num_classes = num_classes

        # 高效的 1D 卷积核
        self.temp_conv1 = nn.Conv1d(
            in_channels=n_bands, out_channels=n_bands * self.out_channels_per_scale,
            kernel_size=65, padding=32, groups=n_bands
        )
        self.temp_conv2 = nn.Conv1d(
            in_channels=n_bands, out_channels=n_bands * self.out_channels_per_scale,
            kernel_size=33, padding=16, groups=n_bands
        )
        self.temp_conv3 = nn.Conv1d(
            in_channels=n_bands, out_channels=n_bands * self.out_channels_per_scale,
            kernel_size=17, padding=8, groups=n_bands
        )

        total_temp_channels = n_bands * self.Z * self.out_channels_per_scale
        self.bn_template = nn.BatchNorm2d(total_temp_channels)
        self.elu = nn.ELU()

        self.spatial_fusion_conv = nn.Conv2d(
            in_channels=total_temp_channels, out_channels=m * n_bands,
            kernel_size=(chans, 1), groups=1
        )
        self.bn_spatial = nn.BatchNorm2d(m * n_bands)

        self.classifier = None

    def _log_variance_layer_fast(self, x):
        x = x.squeeze(2)
        batch_size, channels, T0 = x.shape
        steps = T0 // self.w

        if steps == 0:
            var = torch.var(x, dim=-1, keepdim=True)
            log_var = torch.log(torch.clamp(var, min=1e-6, max=1e6))
            return log_var.view(batch_size, -1)

        x = x[:, :, :steps * self.w].contiguous()
        x = x.view(batch_size, channels, steps, self.w)

        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).sum(dim=-1) / (self.w - 1)

        log_var = torch.log(torch.clamp(var, min=1e-6, max=1e6))
        return log_var.view(batch_size, -1)

    def forward(self, x):
        batch_size, n_bands, chans, T = x.shape

        # 极致一维批处理
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * chans, n_bands, T)

        out1 = self.temp_conv1(x_reshaped)
        out2 = self.temp_conv2(x_reshaped)
        out3 = self.temp_conv3(x_reshaped)

        U_T = torch.cat([out1, out2, out3], dim=1)

        total_temp_channels = n_bands * self.Z * self.out_channels_per_scale
        U_T = U_T.view(batch_size, chans, total_temp_channels, T).permute(0, 2, 1, 3).contiguous()

        U_T = self.elu(self.bn_template(U_T))
        U_TS = self.spatial_fusion_conv(U_T)
        U_TS = self.elu(self.bn_spatial(U_TS))

        U_var = self._log_variance_layer_fast(U_TS)

        if self.classifier is None:
            in_features = U_var.size(1)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            ).to(x.device)

        out = self.classifier(U_var)
        return out


# =========================================================
# 4. AMP MIXED PRECISION TRAINING SYSTEM
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 启动底层算子最优算法搜索
torch.backends.cudnn.benchmark = True

# 滑动窗调整为 25 (对应降采样)
model = BrainprintNet(num_classes=num_classes, chans=C, n_bands=12, m=4, w=25).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 混合精度 Scaler
scaler = torch.cuda.amp.GradScaler()

# 显存连续锁定
train_dataset = TensorDataset(X_train_tensor.to(device).contiguous(), y_train_tensor.to(device))
test_dataset = TensorDataset(X_test_tensor.to(device).contiguous(), y_test_tensor.to(device))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epochs = 150


def evaluate_with_loader(model, loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.cuda.amp.autocast():
                out = model(xb)
            all_outputs.append(out)
    return torch.cat(all_outputs, dim=0)


print("\n🚀 开始（降采样 + 混合精度 AMP）模型训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(xb)
            loss = criterion(out, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # 每 10 轮输出一次表现
    if (epoch + 1) % 10 == 0 or epoch == 0:
        test_logits = evaluate_with_loader(model, test_loader)
        pred = test_logits.argmax(dim=1)
        acc = (pred == y_test_tensor.to(device)).float().mean().item()
        print(f"Epoch {epoch + 1:03d} | Loss {total_loss:.3f} | Test Acc {acc:.4f}")

# =========================================================
# 5. FINAL EVALUATION
# =========================================================
print("\n正在执行最终评估指标计算...")
logits = evaluate_with_loader(model, test_loader)
probs = torch.softmax(logits, dim=1).cpu().numpy()
preds = logits.argmax(dim=1).cpu().numpy()

y_true = y_test_tensor.cpu().numpy()

final_acc = accuracy_score(y_true, preds)
final_rec = recall_score(y_true, preds, average='macro')
final_f1 = f1_score(y_true, preds, average='macro')

y_true_onehot = np.eye(num_classes)[y_true]
y_true_flat = y_true_onehot.ravel()
y_pred_flat = probs.ravel()

fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print("\n" + "=" * 30)
print("       TEST SET METRICS (BrainprintNet - Ultimate Performance)")
print("=" * 30)
print(f"Accuracy  (ACC) : {final_acc:.4f}")
print(f"Recall    (REC) : {final_rec:.4f}")
print(f"F1-Score  (F1)  : {final_f1:.4f}")
print(f"Equal Error Rate (EER): {eer:.4f} ({eer * 100:.2f}%)")
print("=" * 30)

# =========================================================
# 6. 模型权重保存
# =========================================================
model_path = "brainprintnet_speedup_weights.pth"
torch.save(model.state_dict(), model_path)
print(f"✔ 成功保存 BrainprintNet 核心拓扑权重至: '{model_path}'")