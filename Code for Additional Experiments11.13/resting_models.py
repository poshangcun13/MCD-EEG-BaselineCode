# Local recap architectures, snapshotted for this experiment; not official reproductions.
import copy
import torch
from torch import nn
from vit_pytorch import ViT

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

class FBCNet(nn.Module):
    def __init__(self, num_classes, num_bands=5, num_channels=60):
        super().__init__()

        # 1. 时间卷积 (使用 groups 实现独立频带并行处理)
        self.temporal = nn.Sequential(
            nn.Conv2d(num_bands, 16 * num_bands, kernel_size=(1, 64), padding=(0, 32), groups=num_bands),
            nn.BatchNorm2d(16 * num_bands),
            nn.ELU()
        )

        # 2. 空间卷积
        self.spatial = nn.Sequential(
            nn.Conv2d(16 * num_bands, 32 * num_bands, kernel_size=(num_channels, 1), groups=num_bands),
            nn.BatchNorm2d(32 * num_bands),
            nn.ELU()
        )

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * num_bands, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.temporal(x)
        h = self.spatial(h)

        # Variance Pooling: 沿时间轴计算方差
        h = torch.var(h, dim=-1).squeeze(-1)
        h = torch.log(torch.clamp(h, min=1e-6))

        return self.classifier(h)

class SingleBandEEGNet(nn.Module):
    def __init__(self, num_classes, num_channels=60):
        super().__init__()

        # 输入形状: (B, 1, Channels, Time)

        # 1. 时间卷积：在时间轴上提取波形特征
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 2. 空间卷积：融合通道信息（把通道维压缩掉）
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.temporal(x)  # (B, 16, Channels, T)
        h = self.spatial(h)   # (B, 32, 1, T)

        # Variance Pooling: 沿时间轴(dim=-1)计算方差并降维
        h = torch.var(h, dim=-1).squeeze(-1)  # (B, 32)
        h = torch.log(torch.clamp(h, min=1e-6))

        return self.classifier(h)

class EEGViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = ViT(
            image_size=(60, 800),   # 原始时域 2D 排布 (Channels, TimePoints)
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
        self.transformer_blocks = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(depth)])
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
