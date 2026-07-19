import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import butter, lfilter
from torch.utils.data import TensorDataset, DataLoader

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
    在构建FilterBank的同时进行时域降采样，极大缩减后续卷积的计算量
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
        # 跨步切片进行极速降采样
        X_fb[:, b_idx, :, :] = filtered[:, :, ::downsample_factor][:, :, :T_new]

    return X_fb, T_new


# =========================================================
# 2. LOAD DATA (跨任务协议: 记忆阶段 -> 阅读阶段)
# =========================================================
test_mat_path = "../记忆阶段/EEG_hearing_memory.mat"
train_mat_path = "../IQ/EEG_hearing_IQ.mat"

print("=" * 70)
print(f"🚀 SOURCE TASK (Train) : {train_mat_path}")
print(f"🚀 TARGET TASK (Test)  : {test_mat_path}")
print("=" * 70)

# ---- 2.1 加载训练数据（记忆阶段） ----
mat_train = sio.loadmat(train_mat_path)
raw_train = mat_train["EEG_hearing"]  # 形状: (22, 60, 1600, 64)
S_tr, C_tr, T_tr, N_tr = raw_train.shape

# 维度重排与样本展平 -> 形状: (1408, 60, 1600)
X_train_perm = np.transpose(raw_train, (0, 3, 1, 2))
X_train_raw = X_train_perm.reshape(S_tr * N_tr, C_tr, T_tr)
y_train_raw = np.repeat(np.arange(S_tr), N_tr)

# ---- 2.2 加载测试数据（阅读阶段） ----
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

# ---- 2.3 独立滤波器组处理与下采样 ----
DOWNSAMPLE_FACTOR = 2
fs_eeg = 250

X_train_fb, T_net = create_filter_bank_with_downsample(X_train_raw, fs=fs_eeg, downsample_factor=DOWNSAMPLE_FACTOR)
X_test_fb, _      = create_filter_bank_with_downsample(X_test_raw, fs=fs_eeg, downsample_factor=DOWNSAMPLE_FACTOR)

# 注意：BrainprintNet 本身在内部有 BatchNorm 层和 Log-Variance 处理，故无需额外执行时域 Z-score。
# 将处理好的多频带阵列直接转化为计算张量
X_train_tensor = torch.tensor(X_train_fb, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_fb, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test_raw, dtype=torch.long)

print(f"✔ 跨域数据预处理完毕。网络输入形状: {X_train_tensor.shape}")


# =========================================================
# 3. HIGH-PERFORMANCE BRAINPRINTNET IMPLEMENTATION
# =========================================================
class BrainprintNet(nn.Module):
    def __init__(self, num_classes, chans=60, n_bands=12, m=4, w=25):
        super().__init__()
        self.n_bands = n_bands
        self.chans = chans
        self.w = w
        self.Z = 3
        self.out_channels_per_scale = 8
        self.m = m
        self.num_classes = num_classes

        # 高效的一维时间多尺度卷积核
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

# 实例化脑纹结构模型
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


print("\n🚀 开始执行跨任务脑纹识别训练 (Memory -> Recap)...")
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

    # 每 10 轮输出一次全新的目标域表现
    if (epoch + 1) % 10 == 0 or epoch == 0:
        test_logits = evaluate_with_loader(model, test_loader)
        pred = test_logits.argmax(dim=1)
        acc = (pred == y_test_tensor.to(device)).float().mean().item()
        print(f"Epoch {epoch + 1:03d} | Train Loss {total_loss:.3f} | Cross-Task Test Acc {acc:.4f}")

# =========================================================
# 5. FINAL EVALUATION (跨任务泛化能力测试)
# =========================================================
print("\n正在执行目标域 (Reading Task) 最终指标评估...")
logits = evaluate_with_loader(model, test_loader)
probs = torch.softmax(logits, dim=1).cpu().numpy()
preds = logits.argmax(dim=1).cpu().numpy()

y_true = y_test_tensor.cpu().numpy()

# 1. 计算核心多分类泛化指标
final_acc = accuracy_score(y_true, preds)
final_rec = recall_score(y_true, preds, average='macro')
final_f1  = f1_score(y_true, preds, average='macro')

# 2. 计算跨场景身份验证的等错误率 (EER)
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
# 6. 模型权重保存
# =========================================================
save_dir = "./"
model_path = os.path.join(save_dir, "cross_task_mem_to_read_brainprintnet.pth")
torch.save(model.state_dict(), model_path)
print(f"✔ Cross-task model successfully saved to: '{model_path}'")
print("="*45)