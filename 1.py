import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import librosa
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, hamming_loss, jaccard_score
import seaborn as sns
import warnings
import json
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import copy
import itertools
from scipy import stats
import matplotlib.gridspec as gridspec
import time

# Set matplotlib parameters with larger fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14  # Increased from 12
plt.rcParams['axes.labelsize'] = 16  # Increased from 14
plt.rcParams['axes.titlesize'] = 18  # Increased from 16
plt.rcParams['legend.fontsize'] = 13  # Increased from 12
plt.rcParams['xtick.labelsize'] = 13  # Increased from 11
plt.rcParams['ytick.labelsize'] = 13  # Increased from 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 20  # Increased from 18
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlepad'] = 15  # Added padding for titles
plt.rcParams['axes.labelpad'] = 12  # Added padding for labels
warnings.filterwarnings('ignore')

# ==================== 1. SYSTEM CONFIGURATION ====================
class SystemConfig:
    """System Configuration - Five-Class Fault Diagnosis"""

    # Define five fault classes
    FAULT_CLASSES = ['cage', 'inner', 'normal', 'outer', 'roller']
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FAULT_CLASSES)}

    def __init__(self):
        # Data parameters
        self.sample_rate = 25600
        self.signal_length = 2048

        # Time-frequency image parameters
        self.tfi_height = 128
        self.tfi_width = 128
        self.tfi_channels = 3

        # Model parameters - fixed to 5 classes
        self.num_classes = 5  # Five fault classes: cage, inner, normal, outer, roller

        # MC-Net parameters
        self.mc_channels = [32, 64, 128, 256]
        self.kernel_sizes = [3, 5, 7]

        # Mamba parameters
        self.mamba_hidden_dim = 128
        self.mamba_state_dim = 8
        self.num_mamba_layers = 2
        self.num_attention_heads = 2

        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.patience = 20
        self.warmup_epochs = 5

        # Data augmentation
        self.use_augmentation = True
        self.noise_level = 0.01
        self.scale_range = (0.9, 1.1)

        # Visualization parameters
        self.visualization_dpi = 600
        self.save_pdf = True
        self.plot_style = 'seaborn-v0_8-paper'

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Path configuration
        self.checkpoint_dir = 'checkpoints_five_classes'
        self.results_dir = 'results_five_classes'
        self.logs_dir = 'logs_five_classes'
        self.visualizations_dir = 'visualizations_five_classes'
        self.tables_dir = 'tables_five_classes'
        self.create_dirs()

        # Training status
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_dirs(self):
        """Create necessary directories"""
        dirs = [self.checkpoint_dir, self.results_dir, self.logs_dir,
                self.visualizations_dir, self.tables_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def save(self, path='system_config.json'):
        """Save configuration"""
        config_dict = {k: v for k, v in self.__dict__.items()
                       if not k.startswith('_') and k not in ['device', 'start_time']}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def load(self, path='system_config.json'):
        """Load configuration"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

# ==================== 2. TIME-FREQUENCY TRANSFORM MODULE ====================
class TimeFrequencyTransformer(nn.Module):
    """Time-Frequency Image Transformation Module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tfi_height = config.tfi_height
        self.tfi_width = config.tfi_width

        # STFT parameters
        self.n_fft = min(256, config.signal_length // 8)
        self.hop_length = max(64, self.n_fft // 4)
        self.window = torch.hann_window(self.n_fft)

    def stft_transform(self, signal_tensor):
        """STFT transformation"""
        batch_size, seq_len = signal_tensor.shape

        # Ensure signal length is sufficient
        if seq_len < self.n_fft:
            padding = self.n_fft - seq_len
            signal_tensor = F.pad(signal_tensor, (0, padding))
            seq_len = self.n_fft

        # Calculate STFT
        device = signal_tensor.device
        window = self.window.to(device)

        stft_result = torch.stft(
            signal_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=False
        )

        # Calculate magnitude spectrum
        magnitude = torch.abs(stft_result)

        # Convert to decibels
        magnitude_db = 20 * torch.log10(magnitude + 1e-10)

        # Normalize
        magnitude_min = magnitude_db.amin(dim=(1, 2), keepdim=True)
        magnitude_max = magnitude_db.amax(dim=(1, 2), keepdim=True)
        magnitude_normalized = (magnitude_db - magnitude_min) / (magnitude_max - magnitude_min + 1e-10)

        # Resize and convert to three channels
        magnitude_resized = F.interpolate(
            magnitude_normalized.unsqueeze(1),
            size=(self.tfi_height, self.tfi_width),
            mode='bilinear',
            align_corners=False
        )

        magnitude_3channel = magnitude_resized.repeat(1, 3, 1, 1)

        return magnitude_3channel

    def forward(self, signal_tensor):
        """Forward pass"""
        return self.stft_transform(signal_tensor)

# ==================== 3. MULTI-SCALE CONVOLUTION NETWORK ====================
class MultiScaleConvBlock(nn.Module):
    """Multi-scale Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Ensure out_channels is divisible by 3
        out_channels = out_channels // 3 * 3
        branch_channels = out_channels // 3

        # Small kernel branch
        self.conv_small = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Dropout2d(0.05)
        )

        # Medium kernel branch
        self.conv_medium = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Dropout2d(0.05)
        )

        # Large kernel branch
        self.conv_large = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Dropout2d(0.05)
        )

        # Fusion layer
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(1, out_channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(1, out_channels // 4), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.residual(x)

        # Parallel multi-scale convolution
        out1 = self.conv_small(x)
        out2 = self.conv_medium(x)
        out3 = self.conv_large(x)

        # Feature concatenation
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)

        # Channel attention
        attention_weights = self.attention(out)
        out = out * attention_weights

        # Residual connection
        out = out + residual
        out = F.gelu(out)

        return out

class MCNet(nn.Module):
    """MC-Net Feature Extractor"""

    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()

        # Adjust channels to ensure divisibility by 3
        channels = [30, 63, 126, 252]

        # Feature extraction pyramid
        self.layer1 = nn.Sequential(
            MultiScaleConvBlock(in_channels, channels[0]),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            MultiScaleConvBlock(channels[0], channels[1]),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            MultiScaleConvBlock(channels[1], channels[2]),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            MultiScaleConvBlock(channels[2], channels[3]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Feature projection
        self.feature_proj = nn.Linear(channels[3], feature_dim)

    def forward(self, x):
        # Extract multi-scale features
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Extract feature vector
        feature_vector = x4.view(x4.size(0), -1)
        feature_vector = self.feature_proj(feature_vector)

        return feature_vector

# ==================== 4. MAMBA MODULE ====================
class MambaBlock(nn.Module):
    """Mamba Block"""

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Selective scan
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Convolution
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv-1
        )

        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, self.d_inner) * 0.02)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Selective state space
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)

        # 1D convolution
        x1 = x1.transpose(1, 2)
        x1 = self.conv(x1)
        x1 = x1[:, :, :x.shape[1]].transpose(1, 2)

        # Selective scan
        dt = F.softplus(x2)
        A = torch.exp(self.A.T)  # [d_inner, d_state]

        batch_size, seq_len, _ = x1.shape
        state = torch.zeros(batch_size, self.d_inner, A.shape[1], device=x.device)

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            state = state * dt_t + x1[:, t:t+1].transpose(1, 2) * A.unsqueeze(0)
            output = torch.einsum('bik,ik->bi', state, A) + self.D * x1[:, t]
            outputs.append(output.unsqueeze(1))

        x_ssm = torch.cat(outputs, dim=1)
        x_ssm = self.out_proj(x_ssm)
        x = residual + x_ssm

        # MLP
        residual2 = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual2 + x

        return x

class MMamba(nn.Module):
    """MMamba Module"""

    def __init__(self, config, input_dim=128):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.mamba_hidden_dim),
            nn.LayerNorm(config.mamba_hidden_dim),
            nn.GELU()
        )

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(config.mamba_hidden_dim, config.mamba_state_dim)
            for _ in range(config.num_mamba_layers)
        ])

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            config.mamba_hidden_dim,
            config.num_attention_heads,
            batch_first=True,
            dropout=0.1
        )

        # Output projection
        self.output_proj = nn.Linear(config.mamba_hidden_dim, 128)

    def forward(self, x):
        x = self.input_proj(x)

        # Add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Pass through Mamba layers
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)

        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out

        # Global average pooling
        x = x.mean(dim=1)
        x = self.output_proj(x)

        return x

# ==================== 5. FUSION MODULE ====================
class CrossModalAttention(nn.Module):
    """Cross-modal Attention Module"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_proj(output)

        return output.squeeze(1)

class FusionModule(nn.Module):
    """Fusion Module"""

    def __init__(self, config, feature_dim=128):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim

        # Intra-modal feature enhancement
        self.vib_enhance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.acoustic_enhance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(feature_dim, num_heads=4)

        # Feature fusion gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
            nn.Softmax(dim=-1)
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, vib_mc, acoustic_mc, vib_mamba, acoustic_mamba):
        # Intra-modal feature enhancement
        vib_enhanced = self.vib_enhance(vib_mc)
        acoustic_enhanced = self.acoustic_enhance(acoustic_mc)

        # Cross-modal interaction
        vib_cross = self.cross_attention(vib_enhanced.unsqueeze(1),
                                         acoustic_enhanced.unsqueeze(1),
                                         acoustic_enhanced.unsqueeze(1))
        acoustic_cross = self.cross_attention(acoustic_enhanced.unsqueeze(1),
                                              vib_enhanced.unsqueeze(1),
                                              vib_enhanced.unsqueeze(1))

        # Mamba feature enhancement
        vib_mamba_enhanced = self.vib_enhance(vib_mamba)
        acoustic_mamba_enhanced = self.acoustic_enhance(acoustic_mamba)

        # Gated fusion
        all_features = torch.cat([vib_enhanced + vib_cross,
                                  acoustic_enhanced + acoustic_cross,
                                  vib_mamba_enhanced + acoustic_mamba_enhanced], dim=-1)

        gate_weights = self.fusion_gate(all_features)

        # Weighted fusion
        weighted_vib = (vib_enhanced + vib_cross) * gate_weights[:, 0:1]
        weighted_acoustic = (acoustic_enhanced + acoustic_cross) * gate_weights[:, 1:2]
        weighted_mamba = (vib_mamba_enhanced + acoustic_mamba_enhanced) * gate_weights[:, 2:3]

        # Final fusion
        fused = torch.cat([weighted_vib, weighted_acoustic, weighted_mamba], dim=-1)
        output = self.final_fusion(fused)

        return output, {'gate_weights': gate_weights}

# ==================== 6. COMPLETE MC-MMAMBA MODEL ====================
class MC_MMamba(nn.Module):
    """Complete MC-MMamba Model - Five-Class Fault Diagnosis"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = 128

        # Time-frequency transformation
        self.tfi_vib = TimeFrequencyTransformer(config)
        self.tfi_acoustic = TimeFrequencyTransformer(config)

        # Feature extraction
        self.mcnet_vib = MCNet(in_channels=3, feature_dim=self.feature_dim)
        self.mcnet_acoustic = MCNet(in_channels=3, feature_dim=self.feature_dim)

        # Mamba module
        self.mmamba_vib = MMamba(config, input_dim=self.feature_dim)
        self.mmamba_acoustic = MMamba(config, input_dim=self.feature_dim)

        # Fusion module
        self.fusion = FusionModule(config, feature_dim=self.feature_dim)

        # Classifier - outputs 5 classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 5)  # Fixed output 5 classes
        )

        # Adaptive weights
        self.mc_weight = nn.Parameter(torch.tensor(0.5))
        self.mamba_weight = nn.Parameter(torch.tensor(0.5))

        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vib_signal, acoustic_signal):
        # 1. Time-frequency transformation
        vib_tfi = self.tfi_vib(vib_signal)
        acoustic_tfi = self.tfi_acoustic(acoustic_signal)

        # 2. Dual-engine feature extraction
        vib_mc_features = self.mcnet_vib(vib_tfi)
        acoustic_mc_features = self.mcnet_acoustic(acoustic_tfi)

        vib_mamba_features = self.mmamba_vib(vib_mc_features)
        acoustic_mamba_features = self.mmamba_acoustic(acoustic_mc_features)

        # 3. Adaptive weight fusion
        mc_weight = torch.sigmoid(self.mc_weight)
        mamba_weight = torch.sigmoid(self.mamba_weight)

        vib_mc_weighted = vib_mc_features * mc_weight
        vib_mamba_weighted = vib_mamba_features * mamba_weight
        acoustic_mc_weighted = acoustic_mc_features * mc_weight
        acoustic_mamba_weighted = acoustic_mamba_features * mamba_weight

        # 4. Enhanced fusion
        fused_features, attention_info = self.fusion(
            vib_mc_weighted, acoustic_mc_weighted,
            vib_mamba_weighted, acoustic_mamba_weighted
        )

        # 5. Classification
        logits = self.classifier(fused_features)

        outputs = {
            'logits': logits,
            'vib_tfi': vib_tfi,
            'acoustic_tfi': acoustic_tfi,
            'fused_features': fused_features,
            'attention_info': attention_info,
            'mc_weight': mc_weight,
            'mamba_weight': mamba_weight
        }

        return outputs

    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameter_count(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==================== 6.1 消融研究模型变体定义 ====================
class MCNetOnly(nn.Module):
    """MC-Net Only variant (without Mamba)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = 128

        # Time-frequency transformation
        self.tfi_vib = TimeFrequencyTransformer(config)
        self.tfi_acoustic = TimeFrequencyTransformer(config)

        # Feature extraction (MC-Net only)
        self.mcnet_vib = MCNet(in_channels=3, feature_dim=self.feature_dim)
        self.mcnet_acoustic = MCNet(in_channels=3, feature_dim=self.feature_dim)

        # Simplified fusion (without Mamba)
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 5)
        )

    def forward(self, vib_signal, acoustic_signal):
        # Time-frequency transformation
        vib_tfi = self.tfi_vib(vib_signal)
        acoustic_tfi = self.tfi_acoustic(acoustic_signal)

        # Feature extraction
        vib_features = self.mcnet_vib(vib_tfi)
        acoustic_features = self.mcnet_acoustic(acoustic_tfi)

        # Fusion
        combined = torch.cat([vib_features, acoustic_features], dim=1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        return {'logits': logits}

class MambaOnly(nn.Module):
    """Mamba Only variant (without MC-Net)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = 128

        # Time-frequency transformation (simplified)
        self.tfi_vib = TimeFrequencyTransformer(config)
        self.tfi_acoustic = TimeFrequencyTransformer(config)

        # Simple projection to feature dim
        self.vib_proj = nn.Sequential(
            nn.Linear(config.signal_length, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_dim)
        )

        self.acoustic_proj = nn.Sequential(
            nn.Linear(config.signal_length, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_dim)
        )

        # Mamba modules
        self.mmamba_vib = MMamba(config, input_dim=self.feature_dim)
        self.mmamba_acoustic = MMamba(config, input_dim=self.feature_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 5)
        )

    def forward(self, vib_signal, acoustic_signal):
        # Simple projection
        vib_features = self.vib_proj(vib_signal)
        acoustic_features = self.acoustic_proj(acoustic_signal)

        # Mamba processing
        vib_mamba = self.mmamba_vib(vib_features)
        acoustic_mamba = self.mmamba_acoustic(acoustic_features)

        # Fusion
        combined = torch.cat([vib_mamba, acoustic_mamba], dim=1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        return {'logits': logits}

class VibrationOnly(nn.Module):
    """Vibration Only variant (without acoustic)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = 128

        # Time-frequency transformation for vibration only
        self.tfi_vib = TimeFrequencyTransformer(config)

        # MC-Net for vibration
        self.mcnet_vib = MCNet(in_channels=3, feature_dim=self.feature_dim)

        # Mamba for vibration
        self.mmamba_vib = MMamba(config, input_dim=self.feature_dim)

        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 5)
        )

    def forward(self, vib_signal, acoustic_signal):
        # Time-frequency transformation (vibration only)
        vib_tfi = self.tfi_vib(vib_signal)

        # MC-Net features
        vib_mc = self.mcnet_vib(vib_tfi)

        # Mamba features
        vib_mamba = self.mmamba_vib(vib_mc)

        # Fusion (combine MC-Net and Mamba features)
        combined = torch.cat([vib_mc, vib_mamba], dim=1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        return {'logits': logits}

class AcousticOnly(nn.Module):
    """Acoustic Only variant (without vibration)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = 128

        # Time-frequency transformation for acoustic only
        self.tfi_acoustic = TimeFrequencyTransformer(config)

        # MC-Net for acoustic
        self.mcnet_acoustic = MCNet(in_channels=3, feature_dim=self.feature_dim)

        # Mamba for acoustic
        self.mmamba_acoustic = MMamba(config, input_dim=self.feature_dim)

        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 5)
        )

    def forward(self, vib_signal, acoustic_signal):
        # Time-frequency transformation (acoustic only)
        acoustic_tfi = self.tfi_acoustic(acoustic_signal)

        # MC-Net features
        acoustic_mc = self.mcnet_acoustic(acoustic_tfi)

        # Mamba features
        acoustic_mamba = self.mmamba_acoustic(acoustic_mc)

        # Fusion (combine MC-Net and Mamba features)
        combined = torch.cat([acoustic_mc, acoustic_mamba], dim=1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        return {'logits': logits}

# ==================== 7. DATASET CLASS - FIVE CLASSES ====================
class FiveClassDataset(Dataset):
    """Five-Class Fault Dataset"""

    def __init__(self, mat_paths, wav_paths, config=None, is_train=True, augment=False):
        self.mat_paths = mat_paths
        self.wav_paths = wav_paths
        self.config = config or SystemConfig()
        self.is_train = is_train
        self.augment = augment and is_train

        # Define five fault classes
        self.fault_classes = SystemConfig.FAULT_CLASSES
        self.class_to_idx = SystemConfig.CLASS_TO_IDX

        # Extract labels from paths (one of five classes)
        self.labels = self._extract_labels_from_paths(mat_paths)

        # Ensure all labels are in five classes
        self._validate_labels()

        # Label encoding
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        # Set class names
        self.class_names = self.fault_classes

        # Calculate class weights
        self.class_weights = self._compute_class_weights()

    def _extract_labels_from_paths(self, mat_paths):
        """Extract five-class fault labels from paths"""
        labels = []
        for mat_path in mat_paths:
            # Get folder name containing the file
            folder_name = os.path.basename(os.path.dirname(mat_path)).lower()

            # Check if folder name is one of the five classes
            if folder_name in self.fault_classes:
                labels.append(folder_name)
            else:
                # Try to find five-class fault keywords in the path
                path_lower = mat_path.lower()
                for fault_class in self.fault_classes:
                    if fault_class in path_lower:
                        labels.append(fault_class)
                        break
                else:
                    # If not found, guess based on file naming convention
                    basename = os.path.basename(mat_path).lower()
                    if 'cage' in basename:
                        labels.append('cage')
                    elif 'inner' in basename:
                        labels.append('inner')
                    elif 'normal' in basename:
                        labels.append('normal')
                    elif 'outer' in basename:
                        labels.append('outer')
                    elif 'roller' in basename:
                        labels.append('roller')
                    else:
                        # Default to normal class (adjust based on actual situation)
                        labels.append('normal')
                        print(f"Warning: Cannot determine class for file {mat_path}, assigned to normal")

        return labels

    def _validate_labels(self):
        """Validate all labels are in five classes"""
        unique_labels = set(self.labels)
        invalid_labels = unique_labels - set(self.fault_classes)
        if invalid_labels:
            print(f"Warning: Invalid labels found {invalid_labels}, filtered")
            # Filter files with invalid labels
            valid_indices = [i for i, label in enumerate(self.labels) if label in self.fault_classes]
            self.mat_paths = [self.mat_paths[i] for i in valid_indices]
            self.wav_paths = [self.wav_paths[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]

    def _compute_class_weights(self):
        """Calculate class weights"""
        unique, counts = np.unique(self.encoded_labels, return_counts=True)
        total = len(self.encoded_labels)
        weights = total / (len(unique) * counts)
        weights = weights / weights.sum() * len(unique)
        return torch.FloatTensor(weights)

    def _augment_signal(self, signal):
        """Signal augmentation"""
        if not self.augment or np.random.rand() > 0.5:
            return signal

        # Add noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, self.config.noise_level, signal.shape)
            signal = signal + noise

        # Amplitude scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale

        return signal

    def __len__(self):
        return len(self.mat_paths)

    def __getitem__(self, idx):
        # Load vibration signal
        vib_signal = self._load_vibration_signal(self.mat_paths[idx])

        # Load acoustic signal
        acoustic_signal = self._load_acoustic_signal(self.wav_paths[idx])

        # Data augmentation
        if self.augment:
            vib_signal = self._augment_signal(vib_signal)
            acoustic_signal = self._augment_signal(acoustic_signal)

        # Convert to tensors
        vib_tensor = torch.FloatTensor(vib_signal)
        acoustic_tensor = torch.FloatTensor(acoustic_signal)

        # Get label
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)

        return vib_tensor, acoustic_tensor, label

    def _load_vibration_signal(self, file_path):
        """Load vibration signal"""
        try:
            mat_data = scipy.io.loadmat(file_path)

            # Try different data keys
            data_keys = ['vib_data', 'data', 'signal', 'vibration', 'vib']
            vib_data = None

            for key in data_keys:
                if key in mat_data:
                    vib_data = mat_data[key].flatten()
                    break

            # If no standard key found, take first numeric array
            if vib_data is None:
                for key in mat_data.keys():
                    if not key.startswith('__') and np.issubdtype(mat_data[key].dtype, np.number):
                        vib_data = mat_data[key].flatten()
                        break
                else:
                    raise KeyError(f"Vibration data not found: {file_path}")

            vib_data = vib_data.ravel()

            # Standardize length
            target_length = self.config.signal_length

            if len(vib_data) > target_length:
                start_idx = np.random.randint(0, len(vib_data) - target_length)
                vib_data = vib_data[start_idx:start_idx + target_length]
            elif len(vib_data) < target_length:
                padding = target_length - len(vib_data)
                vib_data = np.pad(vib_data, (0, padding), mode='constant')

            # Normalize
            vib_data = (vib_data - np.mean(vib_data)) / (np.std(vib_data) + 1e-8)

            return vib_data.astype(np.float32)
        except Exception as e:
            print(f"Failed to load vibration signal {file_path}: {e}")
            # Return zero signal
            return np.zeros(self.config.signal_length, dtype=np.float32)

    def _load_acoustic_signal(self, file_path):
        """Load acoustic signal"""
        try:
            signal, _ = librosa.load(file_path, sr=self.config.sample_rate, mono=True)

            # Standardize length
            target_length = self.config.signal_length

            if len(signal) > target_length:
                start_idx = np.random.randint(0, len(signal) - target_length)
                signal = signal[start_idx:start_idx + target_length]
            elif len(signal) < target_length:
                padding = target_length - len(signal)
                signal = np.pad(signal, (0, padding), mode='constant')

            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            return signal.astype(np.float32)
        except Exception as e:
            print(f"Failed to load acoustic signal {file_path}: {e}")
            # Return zero signal
            return np.zeros(self.config.signal_length, dtype=np.float32)

# ==================== 8. PROFESSIONAL VISUALIZATION TOOLS - FIVE CLASSES ====================
class FiveClassVisualizer:
    """Five-Class Fault Visualization Tool"""

    def __init__(self, config):
        self.config = config
        self.class_names = SystemConfig.FAULT_CLASSES
        self.colors = plt.cm.Set2(np.linspace(0, 1, 5))

        # Set plot style
        plt.style.use(config.plot_style)

        # Set font parameters with larger fonts
        plt.rcParams['font.size'] = 16  # Increased from 12
        plt.rcParams['axes.labelsize'] = 18  # Increased from 14
        plt.rcParams['axes.titlesize'] = 20  # Increased from 16
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = 14  # Increased from 11
        plt.rcParams['xtick.labelsize'] = 14  # Increased from 11
        plt.rcParams['ytick.labelsize'] = 14  # Increased from 11
        plt.rcParams['figure.titlesize'] = 22  # Increased from 18
        plt.rcParams['figure.titleweight'] = 'bold'
        plt.rcParams['axes.titlepad'] = 18  # Added padding for titles
        plt.rcParams['axes.labelpad'] = 14  # Added padding for labels

    def create_training_summary(self, trainer, save_path=None):
        """Create training summary plot with larger fonts"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 16), dpi=self.config.visualization_dpi)  # Increased figure size
        axes = axes.flatten()

        epochs = range(1, len(trainer.train_losses) + 1)

        # 1. Loss curves
        axes[0].plot(epochs, trainer.train_losses, 'b-', label='Training Loss',
                     linewidth=3.0, alpha=0.8, marker='o', markersize=7, markevery=5)
        axes[0].plot(epochs, trainer.val_losses, 'r-', label='Validation Loss',
                     linewidth=3.0, alpha=0.8, marker='s', markersize=7, markevery=5)
        axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=18)
        axes[0].set_ylabel('Loss Value', fontweight='bold', fontsize=18)
        axes[0].set_title('Training & Validation Loss Curves', fontweight='bold', fontsize=20, pad=15)
        axes[0].legend(fontsize=14, loc='best')
        axes[0].grid(True, alpha=0.3, linewidth=0.5)
        axes[0].tick_params(labelsize=14)

        # 2. Accuracy curves
        axes[1].plot(epochs, trainer.train_accs, 'b-', label='Training Accuracy',
                     linewidth=3.0, alpha=0.8, marker='o', markersize=7, markevery=5)
        axes[1].plot(epochs, trainer.val_accs, 'r-', label='Validation Accuracy',
                     linewidth=3.0, alpha=0.8, marker='s', markersize=7, markevery=5)
        axes[1].axhline(y=trainer.best_accuracy, color='g', linestyle='--',
                        linewidth=2.5, label=f'Best: {trainer.best_accuracy:.1f}%')
        axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=18)
        axes[1].set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
        axes[1].set_title('Training & Validation Accuracy Curves', fontweight='bold', fontsize=20, pad=15)
        axes[1].legend(fontsize=14, loc='best')
        axes[1].grid(True, alpha=0.3, linewidth=0.5)
        axes[1].set_ylim([0, 105])
        axes[1].tick_params(labelsize=14)

        # 3. F1-score curves
        axes[2].plot(epochs, trainer.train_f1s, 'b-', label='Training F1-Score',
                     linewidth=3.0, alpha=0.8, marker='o', markersize=7, markevery=5)
        axes[2].plot(epochs, trainer.val_f1s, 'r-', label='Validation F1-Score',
                     linewidth=3.0, alpha=0.8, marker='s', markersize=7, markevery=5)
        axes[2].set_xlabel('Epoch', fontweight='bold', fontsize=18)
        axes[2].set_ylabel('F1-Score', fontweight='bold', fontsize=18)
        axes[2].set_title('Training & Validation F1-Score Curves', fontweight='bold', fontsize=20, pad=15)
        axes[2].legend(fontsize=14, loc='best')
        axes[2].grid(True, alpha=0.3, linewidth=0.5)
        axes[2].set_ylim([0, 1.05])
        axes[2].tick_params(labelsize=14)

        # 4. Learning rate curve
        if hasattr(trainer, 'learning_rates') and trainer.learning_rates:
            axes[3].plot(range(1, len(trainer.learning_rates) + 1), trainer.learning_rates,
                         'g-', linewidth=3.0, marker='^', markersize=7, markevery=5)
            axes[3].set_xlabel('Epoch', fontweight='bold', fontsize=18)
            axes[3].set_ylabel('Learning Rate', fontweight='bold', fontsize=18)
            axes[3].set_title('Learning Rate Schedule', fontweight='bold', fontsize=20, pad=15)
            axes[3].grid(True, alpha=0.3, linewidth=0.5)
            axes[3].set_yscale('log')
            axes[3].tick_params(labelsize=14)

        # 5. Model information
        model = trainer.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = f"MC-MMamba Five-Class Fault Diagnosis\n"
        model_info += f"Total Parameters: {total_params:,}\n"
        model_info += f"Trainable Parameters: {trainable_params:,}\n"
        model_info += f"Best Validation Accuracy: {trainer.best_accuracy:.2f}%\n"
        model_info += f"Best Epoch: {trainer.best_epoch}\n"
        model_info += f"Fault Classes: {', '.join(self.class_names)}"

        axes[4].text(0.5, 0.5, model_info, ha='center', va='center', fontsize=16,
                     linespacing=1.8, transform=axes[4].transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[4].set_title('Model Information', fontweight='bold', fontsize=20, pad=15)
        axes[4].axis('off')

        # 6. Class distribution (training set)
        if hasattr(trainer, 'train_loader'):
            try:
                dataset = trainer.train_loader.dataset
                if hasattr(dataset, 'encoded_labels'):
                    unique, counts = np.unique(dataset.encoded_labels, return_counts=True)
                    bars = axes[5].bar(range(len(unique)), counts, color=self.colors[:len(unique)],
                                       edgecolor='black', linewidth=1.8)
                    axes[5].set_xlabel('Fault Class', fontweight='bold', fontsize=18)
                    axes[5].set_ylabel('Sample Count', fontweight='bold', fontsize=18)
                    axes[5].set_title('Training Set Class Distribution', fontweight='bold', fontsize=20, pad=15)
                    axes[5].set_xticks(range(len(unique)))
                    axes[5].set_xticklabels(self.class_names[:len(unique)], rotation=45, fontsize=14, ha='right')
                    axes[5].grid(True, alpha=0.3, linewidth=0.5, axis='y')
                    axes[5].tick_params(labelsize=14)

                    # Add values on bars with larger font
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        axes[5].text(bar.get_x() + bar.get_width()/2., height + 1,
                                     f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            except:
                axes[5].axis('off')
        else:
            axes[5].axis('off')

        plt.suptitle('MC-MMamba Five-Class Fault Diagnosis Training Summary',
                     fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)  # Increased padding

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'training_summary_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Training summary saved to: {save_path}")

    def create_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Create five-class confusion matrix with larger fonts"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(14, 12), dpi=self.config.visualization_dpi)  # Increased figure size

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Value', fontweight='bold', fontsize=16)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               ylabel='True Label',
               xlabel='Predicted Label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2%})",
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')  # Increased font size

        ax.set_title('Five-Class Fault Confusion Matrix', fontweight='bold', fontsize=20, pad=20)
        ax.tick_params(labelsize=14)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'confusion_matrix_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Confusion matrix saved to: {save_path}")

        return cm

    def create_class_performance_chart(self, metrics_dict, save_path=None):
        """Create class-wise performance comparison chart with larger fonts"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=self.config.visualization_dpi)  # Increased figure size

        # Precision
        bars1 = axes[0].bar(range(5), metrics_dict['precision_per_class'],
                            color=self.colors, edgecolor='black', linewidth=2.0)
        axes[0].set_xlabel('Fault Class', fontweight='bold', fontsize=18)
        axes[0].set_ylabel('Precision', fontweight='bold', fontsize=18)
        axes[0].set_title('Class-wise Precision Comparison', fontweight='bold', fontsize=20, pad=15)
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels(self.class_names, rotation=45, fontsize=14, ha='right')
        axes[0].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[0].set_ylim([0, 1.05])
        axes[0].tick_params(labelsize=14)

        # Add values on bars with larger font
        for bar, value in zip(bars1, metrics_dict['precision_per_class']):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Recall
        bars2 = axes[1].bar(range(5), metrics_dict['recall_per_class'],
                            color=self.colors, edgecolor='black', linewidth=2.0)
        axes[1].set_xlabel('Fault Class', fontweight='bold', fontsize=18)
        axes[1].set_ylabel('Recall', fontweight='bold', fontsize=18)
        axes[1].set_title('Class-wise Recall Comparison', fontweight='bold', fontsize=20, pad=15)
        axes[1].set_xticks(range(5))
        axes[1].set_xticklabels(self.class_names, rotation=45, fontsize=14, ha='right')
        axes[1].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[1].set_ylim([0, 1.05])
        axes[1].tick_params(labelsize=14)

        # Add values on bars with larger font
        for bar, value in zip(bars2, metrics_dict['recall_per_class']):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # F1-score
        bars3 = axes[2].bar(range(5), metrics_dict['f1_per_class'],
                            color=self.colors, edgecolor='black', linewidth=2.0)
        axes[2].set_xlabel('Fault Class', fontweight='bold', fontsize=18)
        axes[2].set_ylabel('F1-Score', fontweight='bold', fontsize=18)
        axes[2].set_title('Class-wise F1-Score Comparison', fontweight='bold', fontsize=20, pad=15)
        axes[2].set_xticks(range(5))
        axes[2].set_xticklabels(self.class_names, rotation=45, fontsize=14, ha='right')
        axes[2].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[2].set_ylim([0, 1.05])
        axes[2].tick_params(labelsize=14)

        # Add values on bars with larger font
        for bar, value in zip(bars3, metrics_dict['f1_per_class']):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.suptitle('Five-Class Fault Diagnosis Performance Comparison',
                     fontsize=22, fontweight='bold', y=1.05)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'class_performance_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Class performance chart saved to: {save_path}")

    def create_roc_curves(self, y_true, y_prob, save_path=None):
        """Create ROC curves (five classes) with larger fonts"""
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(5))

        fig, ax = plt.subplots(figsize=(14, 12), dpi=self.config.visualization_dpi)  # Increased figure size

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(5):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=self.colors[i], lw=3.0,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.5, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontweight='bold', fontsize=18)
        ax.set_ylabel('True Positive Rate (TPR)', fontweight='bold', fontsize=18)
        ax.set_title('Five-Class Fault ROC Curves', fontweight='bold', fontsize=20, pad=15)
        ax.legend(loc="lower right", fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=14)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'roc_curves_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ ROC curves saved to: {save_path}")

        return roc_auc

    def create_feature_visualization(self, features, labels, save_path=None):
        """Create feature visualization plot (t-SNE and PCA) with larger fonts"""
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=self.config.visualization_dpi)  # Increased figure size

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_tsne = tsne.fit_transform(features_sample)

        scatter1 = axes[0].scatter(features_tsne[:, 0], features_tsne[:, 1],
                                   c=labels_sample, cmap='tab10', s=60, alpha=0.8,
                                   edgecolors='k', linewidth=0.5)

        axes[0].set_xlabel('t-SNE Component 1', fontweight='bold', fontsize=18)
        axes[0].set_ylabel('t-SNE Component 2', fontweight='bold', fontsize=18)
        axes[0].set_title('Feature Space Visualization (t-SNE)', fontweight='bold', fontsize=20, pad=15)
        axes[0].grid(True, alpha=0.3, linewidth=0.5)
        axes[0].tick_params(labelsize=14)

        # Create legend with larger font
        legend_elements1 = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=plt.cm.tab10(i/10),
                                       markersize=12, label=self.class_names[i])
                            for i in range(min(5, len(self.class_names)))]
        axes[0].legend(handles=legend_elements1, loc="upper right", fontsize=12)

        # PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_sample)

        scatter2 = axes[1].scatter(features_pca[:, 0], features_pca[:, 1],
                                   c=labels_sample, cmap='tab10', s=60, alpha=0.8,
                                   edgecolors='k', linewidth=0.5)

        explained_var = pca.explained_variance_ratio_
        axes[1].set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)',
                           fontweight='bold', fontsize=18)
        axes[1].set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)',
                           fontweight='bold', fontsize=18)
        axes[1].set_title('Feature Space Visualization (PCA)', fontweight='bold', fontsize=20, pad=15)
        axes[1].grid(True, alpha=0.3, linewidth=0.5)
        axes[1].tick_params(labelsize=14)

        # Create legend with larger font
        legend_elements2 = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=plt.cm.tab10(i/10),
                                       markersize=12, label=self.class_names[i])
                            for i in range(min(5, len(self.class_names)))]
        axes[1].legend(handles=legend_elements2, loc="upper right", fontsize=12)

        plt.suptitle('Five-Class Fault Feature Space Visualization',
                     fontsize=22, fontweight='bold', y=1.05)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'feature_visualization_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Feature visualization saved to: {save_path}")

# ==================== 9. FIVE-CLASS FAULT EVALUATOR ====================
class FiveClassEvaluator:
    """Five-Class Fault Evaluator"""

    def __init__(self, config):
        self.config = config
        self.class_names = SystemConfig.FAULT_CLASSES
        self.results = {}

    def compute_all_metrics(self, y_true, y_pred, y_prob=None):
        """Compute all evaluation metrics"""
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        if y_prob is not None:
            # AUC metrics
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(5))

            auc_scores = []
            for i in range(5):
                if len(np.unique(y_true_bin[:, i])) > 1:
                    auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    auc_scores.append(auc_score)

            metrics['auc_per_class'] = auc_scores if auc_scores else [0] * 5
            metrics['mean_auc'] = np.mean(auc_scores) if auc_scores else 0

        # Other statistical metrics
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        self.results = metrics
        return metrics

    def print_detailed_report(self, y_true, y_pred):
        """Print detailed evaluation report"""
        print("\n" + "="*80)
        print("MC-MMAMBA FIVE-CLASS FAULT DIAGNOSIS EVALUATION REPORT")
        print("="*80)

        print(f"\n📊 OVERALL PERFORMANCE METRICS:")
        print(f"   Accuracy:                {self.results.get('accuracy', 0):.4f}")
        print(f"   Precision (Weighted):    {self.results.get('precision', 0):.4f}")
        print(f"   Recall (Weighted):       {self.results.get('recall', 0):.4f}")
        print(f"   F1-Score (Weighted):     {self.results.get('f1', 0):.4f}")
        print(f"   Cohen's Kappa:           {self.results.get('kappa', 0):.4f}")
        print(f"   Matthews Correlation:    {self.results.get('mcc', 0):.4f}")

        if 'mean_auc' in self.results:
            print(f"   Mean AUC:                {self.results.get('mean_auc', 0):.4f}")

        print(f"\n📋 PER-CLASS PERFORMANCE METRICS:")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
        print("-" * 60)

        for i, name in enumerate(self.class_names):
            precision = self.results['precision_per_class'][i] if i < len(self.results['precision_per_class']) else 0
            recall = self.results['recall_per_class'][i] if i < len(self.results['recall_per_class']) else 0
            f1 = self.results['f1_per_class'][i] if i < len(self.results['f1_per_class']) else 0
            auc = self.results['auc_per_class'][i] if i < len(self.results.get('auc_per_class', [])) else 0

            print(f"{name:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {auc:<12.4f}")

        print("\n" + "="*80)
        print("CLASSIFICATION REPORT:")
        print("="*80)
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=4))

    def save_results_to_csv(self, save_path=None):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return

        if save_path is None:
            save_path = os.path.join(self.config.tables_dir,
                                     f'evaluation_results_{self.config.start_time}.csv')

        # Create results DataFrame
        results_data = []

        # Overall metrics
        results_data.append({
            'Metric': 'Accuracy',
            'Value': f"{self.results.get('accuracy', 0):.4f}",
            'Class': 'Overall'
        })
        results_data.append({
            'Metric': 'Precision',
            'Value': f"{self.results.get('precision', 0):.4f}",
            'Class': 'Overall'
        })
        results_data.append({
            'Metric': 'Recall',
            'Value': f"{self.results.get('recall', 0):.4f}",
            'Class': 'Overall'
        })
        results_data.append({
            'Metric': 'F1-Score',
            'Value': f"{self.results.get('f1', 0):.4f}",
            'Class': 'Overall'
        })
        results_data.append({
            'Metric': 'Cohen\'s Kappa',
            'Value': f"{self.results.get('kappa', 0):.4f}",
            'Class': 'Overall'
        })
        results_data.append({
            'Metric': 'Matthews Correlation',
            'Value': f"{self.results.get('mcc', 0):.4f}",
            'Class': 'Overall'
        })

        if 'mean_auc' in self.results:
            results_data.append({
                'Metric': 'Mean AUC',
                'Value': f"{self.results.get('mean_auc', 0):.4f}",
                'Class': 'Overall'
            })

        # Per-class metrics
        if 'precision_per_class' in self.results:
            for i, name in enumerate(self.class_names):
                if i < len(self.results['precision_per_class']):
                    results_data.append({
                        'Metric': 'Precision',
                        'Value': f"{self.results['precision_per_class'][i]:.4f}",
                        'Class': name
                    })
                    results_data.append({
                        'Metric': 'Recall',
                        'Value': f"{self.results['recall_per_class'][i]:.4f}",
                        'Class': name
                    })
                    results_data.append({
                        'Metric': 'F1-Score',
                        'Value': f"{self.results['f1_per_class'][i]:.4f}",
                        'Class': name
                    })
                    if 'auc_per_class' in self.results and i < len(self.results['auc_per_class']):
                        results_data.append({
                            'Metric': 'AUC',
                            'Value': f"{self.results['auc_per_class'][i]:.4f}",
                            'Class': name
                        })

        df = pd.DataFrame(results_data)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"✓ Evaluation results saved to: {save_path}")

        return df

# ==================== 9.1 EXTENDED EVALUATION METRICS CLASS ====================
class EnhancedFiveClassEvaluator(FiveClassEvaluator):
    """Enhanced Five-Class Fault Evaluator (with trustworthiness metrics)"""

    def __init__(self, config):
        super().__init__(config)

    def compute_trustworthiness_metrics(self, y_true, y_pred, y_prob):
        """Compute trustworthiness-related metrics"""

        metrics = {}

        # 1. Basic confidence metrics
        confidences = np.max(y_prob, axis=1)
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['median_confidence'] = float(np.median(confidences))
        metrics['std_confidence'] = float(np.std(confidences))

        # 2. High confidence ratio
        metrics['high_confidence_rate_80'] = float(np.sum(confidences >= 0.8) / len(confidences))
        metrics['high_confidence_rate_90'] = float(np.sum(confidences >= 0.9) / len(confidences))

        # 3. Confidence-accuracy relationship
        correct_mask = (y_pred == y_true)
        metrics['confidence_correct_mean'] = float(np.mean(confidences[correct_mask]) if np.sum(correct_mask) > 0 else 0)
        metrics['confidence_wrong_mean'] = float(np.mean(confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0)

        # 4. Per-class confidence
        confidence_per_class = []
        for i in range(len(self.class_names)):
            mask = y_true == i
            if np.sum(mask) > 0:
                confidence_per_class.append(float(np.mean(confidences[mask])))
            else:
                confidence_per_class.append(0.0)
        metrics['confidence_per_class'] = confidence_per_class

        # 5. Prediction entropy (uncertainty)
        epsilon = 1e-10
        entropy = -np.sum(y_prob * np.log(y_prob + epsilon), axis=1)
        metrics['mean_entropy'] = float(np.mean(entropy))
        metrics['entropy_per_class'] = []
        for i in range(len(self.class_names)):
            mask = y_true == i
            if np.sum(mask) > 0:
                metrics['entropy_per_class'].append(float(np.mean(entropy[mask])))
            else:
                metrics['entropy_per_class'].append(0.0)

        # 6. Calibration error
        correct_predictions = (y_pred == y_true).astype(int)

        if len(np.unique(correct_predictions)) > 1:
            try:
                prob_true, prob_pred = calibration_curve(correct_predictions, confidences,
                                                         n_bins=10, strategy='uniform')
                ece = np.sum(np.abs(prob_true - prob_pred) *
                             (np.histogram(confidences, bins=10)[0] / len(confidences)))
                metrics['expected_calibration_error'] = float(ece)
            except Exception as e:
                # Use simplified ECE calculation
                bin_edges = np.linspace(0, 1, 11)
                bin_indices = np.digitize(confidences, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, 9)

                ece = 0
                for i in range(10):
                    mask = bin_indices == i
                    if np.sum(mask) > 0:
                        bin_mean_prob = np.mean(confidences[mask])
                        bin_accuracy = np.mean(correct_predictions[mask])
                        ece += np.abs(bin_mean_prob - bin_accuracy) * (np.sum(mask) / len(confidences))
                metrics['expected_calibration_error'] = float(ece)
        else:
            metrics['expected_calibration_error'] = 0.0

        # 7. Modality consistency
        metrics['modality_consistency'] = 0.0

        return metrics

    def compute_all_metrics_with_trustworthiness(self, y_true, y_pred, y_prob):
        """Compute all metrics including trustworthiness"""

        # Basic classification metrics
        base_metrics = self.compute_all_metrics(y_true, y_pred, y_prob)

        # Trustworthiness metrics
        trust_metrics = self.compute_trustworthiness_metrics(y_true, y_pred, y_prob)

        # Combine metrics
        all_metrics = {**base_metrics, **trust_metrics}

        # Compute overall trustworthiness score
        all_metrics['overall_trustworthiness_score'] = self._compute_overall_trust_score(all_metrics)

        self.results = all_metrics
        return all_metrics

    def _compute_overall_trust_score(self, metrics):
        """Compute overall trustworthiness score"""

        # Weight settings
        weights = {
            'accuracy': 0.25,
            'f1': 0.20,
            'mean_confidence': 0.15,
            'high_confidence_rate_80': 0.10,
            'confidence_correct_mean': 0.10,
            'expected_calibration_error': 0.10,
            'mean_entropy': 0.10
        }

        # Normalize each metric (0-100 scale)
        normalized_scores = {}

        # Accuracy (0-1 -> 0-100)
        normalized_scores['accuracy'] = metrics.get('accuracy', 0) * 100

        # F1-score (0-1 -> 0-100)
        normalized_scores['f1'] = metrics.get('f1', 0) * 100

        # Average confidence (0-1 -> 0-100)
        normalized_scores['mean_confidence'] = metrics.get('mean_confidence', 0) * 100

        # High confidence ratio (0-1 -> 0-100)
        normalized_scores['high_confidence_rate_80'] = metrics.get('high_confidence_rate_80', 0) * 100

        # Correct prediction average confidence (0-1 -> 0-100)
        normalized_scores['confidence_correct_mean'] = metrics.get('confidence_correct_mean', 0) * 100

        # Calibration error (smaller is better, 0-0.5 -> 100-0)
        ece = metrics.get('expected_calibration_error', 0.5)
        normalized_scores['expected_calibration_error'] = max(0, 100 - ece * 200)

        # Average entropy (smaller is better, 0-log(5)≈1.61 -> 100-0)
        max_entropy = np.log(5)
        entropy = metrics.get('mean_entropy', max_entropy)
        normalized_scores['mean_entropy'] = max(0, 100 - (entropy / max_entropy) * 100)

        # Compute weighted total score
        total_score = 0
        for key, weight in weights.items():
            if key in normalized_scores:
                total_score += normalized_scores[key] * weight

        return float(total_score)

# ==================== 10. FIVE-CLASS FAULT TRAINER ====================
class FiveClassTrainer:
    """Five-Class Fault Trainer"""

    def __init__(self, model, config, class_weights=None):
        self.model = model
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.warmup_scheduler = None
        if config.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: min(1.0, (epoch + 1) / config.warmup_epochs)
            )

        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=1e-6
        )

        # Training records
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []
        self.learning_rates = []

        # Best model tracking
        self.best_accuracy = 0
        self.best_model_state = None
        self.best_epoch = 0

        # Data loader references
        self.train_loader = None
        self.val_loader = None

    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for batch_idx, (vib, audio, labels) in enumerate(pbar):
            vib = vib.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(vib, audio)
            logits = outputs['logits']

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Regularization loss
            if 'mc_weight' in outputs and 'mamba_weight' in outputs:
                mc_weight = outputs['mc_weight']
                mamba_weight = outputs['mamba_weight']
                reg_loss = 0.1 * (torch.abs(mc_weight - 0.5) + torch.abs(mamba_weight - 0.5))
                loss = loss + reg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0.0

        return avg_loss, accuracy, f1

    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for vib, audio, labels in tqdm(val_loader, desc='Validation'):
                vib = vib.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(vib, audio)
                logits = outputs['logits']

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0.0

        return avg_loss, accuracy, f1, all_preds, all_labels

    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.train_loader = train_loader
        self.val_loader = val_loader

        print(f"Starting MC-MMamba Five-Class Fault Diagnosis Training")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Fault classes: {SystemConfig.FAULT_CLASSES}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")

        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_f1s.append(train_f1)

            # Validation
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            if epoch < self.config.warmup_epochs and self.warmup_scheduler:
                self.warmup_scheduler.step()
            elif epoch >= self.config.warmup_epochs:
                self.main_scheduler.step()

            # Print results
            print(f"Learning rate: {current_lr:.6f}")
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, F1: {train_f1:.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1: {val_f1:.4f}")

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ Best model saved, accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

            # Print classification report every 10 epochs
            if (epoch + 1) % 10 == 0 and len(val_labels) > 0 and len(np.unique(val_labels)) > 1:
                print("\nValidation Classification Report:")
                target_names = SystemConfig.FAULT_CLASSES
                print(classification_report(val_labels, val_preds, target_names=target_names, digits=4))

            # Early stopping check
            if patience_counter >= self.config.patience:
                print(f"⏹️ Early stopping triggered, validation accuracy not improved for {self.config.patience} epochs")
                break

        # Training completed
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}% (Epoch {self.best_epoch})")

        # Save final model
        self.save_checkpoint(self.config.num_epochs, is_final=True)

        return self.best_accuracy

    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s,
            'learning_rates': self.learning_rates,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'config': self.config.__dict__
        }

        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        elif is_final:
            path = os.path.join(self.config.checkpoint_dir, 'final_model.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        self.train_f1s = checkpoint['train_f1s']
        self.val_f1s = checkpoint['val_f1s']
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_epoch = checkpoint.get('best_epoch', 0)

        return checkpoint['epoch']

# ==================== 11. EXPLAINABILITY ANALYSIS MODULE ====================
class ExplainabilityAnalyzer:
    """Explainability Analysis Module - Enhanced Model Trustworthiness and Diagnostic Transparency"""

    def __init__(self, config):
        self.config = config
        self.class_names = SystemConfig.FAULT_CLASSES
        self.colors = plt.cm.Set2(np.linspace(0, 1, 5))

        # Set plot style
        plt.style.use(config.plot_style)

        # Set font parameters with larger fonts
        plt.rcParams['font.size'] = 16  # Increased from 12
        plt.rcParams['axes.labelsize'] = 18  # Increased from 14
        plt.rcParams['axes.titlesize'] = 20  # Increased from 16
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = 14  # Increased from 11
        plt.rcParams['xtick.labelsize'] = 14  # Increased from 11
        plt.rcParams['ytick.labelsize'] = 14  # Increased from 11
        plt.rcParams['figure.titlesize'] = 22  # Increased from 18
        plt.rcParams['figure.titleweight'] = 'bold'
        plt.rcParams['axes.titlepad'] = 18  # Added padding for titles
        plt.rcParams['axes.labelpad'] = 14  # Added padding for labels

    def create_grad_cam_heatmaps(self, model, dataloader, num_samples=5, save_path=None):
        """Create Grad-CAM heatmaps - Visualize model attention regions with larger fonts"""
        model.eval()

        # Register hooks to get gradients
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]

        # Get the last convolutional layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'layer4' in name:
                target_layer = module
                break

        if target_layer is None:
            print("No suitable convolutional layer found for Grad-CAM")
            return

        # Register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # Collect samples, ensure coverage of all classes
        samples_by_class = {i: [] for i in range(5)}

        for vib, audio, labels in dataloader:
            for i in range(len(labels)):
                label = labels[i].item()
                if len(samples_by_class[label]) < 1:  # At least one sample per class
                    samples_by_class[label].append((
                        vib[i:i+1], audio[i:i+1], labels[i:i+1]
                    ))

            # Check if we have collected samples for all classes
            if all(len(samples) >= 1 for samples in samples_by_class.values()):
                break

        # Create visualization with larger figure size
        fig, axes = plt.subplots(5, 5, figsize=(22, 22), dpi=self.config.visualization_dpi)

        for class_idx, class_name in enumerate(self.class_names):
            if not samples_by_class[class_idx]:
                continue

            vib, audio, label = samples_by_class[class_idx][0]
            vib = vib.to(self.config.device)
            audio = audio.to(self.config.device)
            label = label.to(self.config.device)

            # Forward pass
            model.zero_grad()
            outputs = model(vib, audio)
            logits = outputs['logits']

            # Get prediction probabilities
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            true_class = label.item()

            # Backward pass to get gradients
            model.zero_grad()
            one_hot_output = torch.zeros_like(logits)
            one_hot_output[0, pred_class] = 1
            logits.backward(gradient=one_hot_output, retain_graph=True)

            # Compute Grad-CAM
            if 'value' in activations and 'value' in gradients:
                activations_val = activations['value'].detach().cpu().numpy()[0]
                gradients_val = gradients['value'].detach().cpu().numpy()[0]

                weights = np.mean(gradients_val, axis=(1, 2))
                cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

                for i, w in enumerate(weights):
                    cam += w * activations_val[i]

                # ReLU activation
                cam = np.maximum(cam, 0)
                cam = cam / (cam.max() + 1e-10)

                # Visualization
                vib_tfi = outputs['vib_tfi'].detach().cpu().numpy()[0].mean(0)
                acoustic_tfi = outputs['acoustic_tfi'].detach().cpu().numpy()[0].mean(0)

                # Original time-frequency images
                axes[class_idx, 0].imshow(vib_tfi, aspect='auto', cmap='viridis')
                axes[class_idx, 0].set_title(f'Vibration TFI\nTrue: {class_name}', fontsize=12)
                axes[class_idx, 0].axis('off')

                axes[class_idx, 1].imshow(acoustic_tfi, aspect='auto', cmap='viridis')
                axes[class_idx, 1].set_title(f'Acoustic TFI\nPred: {self.class_names[pred_class]}', fontsize=12)
                axes[class_idx, 1].axis('off')

                # Heatmap overlay
                axes[class_idx, 2].imshow(vib_tfi, aspect='auto', cmap='gray')
                axes[class_idx, 2].imshow(cam, aspect='auto', cmap='jet', alpha=0.5)
                axes[class_idx, 2].set_title('Grad-CAM Heatmap\n(Vibration)', fontsize=12)
                axes[class_idx, 2].axis('off')

                # Attention weights visualization
                if 'attention_info' in outputs:
                    gate_weights = outputs['attention_info']['gate_weights'].detach().cpu().numpy()[0]
                    axes[class_idx, 3].bar(['Vib-MC', 'Audio-MC', 'Mamba'], gate_weights,
                                           color=['blue', 'orange', 'green'])
                    axes[class_idx, 3].set_title('Fusion Gate Weights', fontsize=12)
                    axes[class_idx, 3].set_ylim([0, 1])
                    axes[class_idx, 3].grid(True, alpha=0.3)
                    axes[class_idx, 3].tick_params(labelsize=10)

                # Confidence distribution
                probs_np = probs.detach().cpu().numpy()[0]
                axes[class_idx, 4].barh(self.class_names, probs_np, color=self.colors)
                axes[class_idx, 4].set_xlim([0, 1])
                axes[class_idx, 4].set_title(f'Diagnosis Confidence\n{probs_np[pred_class]:.2%}', fontsize=12)
                axes[class_idx, 4].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
                axes[class_idx, 4].tick_params(labelsize=10)

        plt.suptitle('MC-MMamba Explainability Analysis - Grad-CAM Heatmaps (Five-Class Faults)',
                     fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'grad_cam_heatmaps_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Grad-CAM heatmaps saved to: {save_path}")

    def create_confidence_heatmap(self, y_true, y_pred, y_prob, save_path=None):
        """Create confidence heatmap - Visualize diagnosis trustworthiness with larger fonts"""

        # Compute confidence for each sample
        confidences = np.max(y_prob, axis=1)

        # Create heatmap grid
        n_classes = len(self.class_names)
        heatmap = np.zeros((n_classes, 5))  # 5 confidence intervals

        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for true_class in range(n_classes):
            mask = y_true == true_class
            if np.sum(mask) > 0:
                for i in range(5):
                    bin_mask = (confidences[mask] >= confidence_bins[i]) & \
                               (confidences[mask] < confidence_bins[i+1])
                    if np.sum(mask) > 0:
                        heatmap[true_class, i] = np.sum(bin_mask) / np.sum(mask)

        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=self.config.visualization_dpi)  # Increased figure size

        # 1. Confidence heatmap
        im = axes[0, 0].imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        axes[0, 0].set_xlabel('Confidence Interval', fontweight='bold', fontsize=18)
        axes[0, 0].set_ylabel('Fault Class', fontweight='bold', fontsize=18)
        axes[0, 0].set_title('Diagnosis Confidence Heatmap', fontweight='bold', fontsize=20, pad=15)
        axes[0, 0].set_xticks(range(5))
        axes[0, 0].set_xticklabels(['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                                   rotation=45, fontsize=14)
        axes[0, 0].set_yticks(range(n_classes))
        axes[0, 0].set_yticklabels(self.class_names, fontsize=14)
        cbar = plt.colorbar(im, ax=axes[0, 0])
        cbar.set_label('Normalized Count', fontweight='bold', fontsize=14)
        axes[0, 0].tick_params(labelsize=14)

        # Add value labels with larger font
        for i in range(n_classes):
            for j in range(5):
                text = axes[0, 0].text(j, i, f'{heatmap[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')

        # 2. Confidence distribution histogram
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(confidences), color='red', linestyle='--',
                           linewidth=2.5, label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 1].axvline(x=np.median(confidences), color='green', linestyle='--',
                           linewidth=2.5, label=f'Median: {np.median(confidences):.3f}')
        axes[0, 1].set_xlabel('Confidence', fontweight='bold', fontsize=18)
        axes[0, 1].set_ylabel('Sample Count', fontweight='bold', fontsize=18)
        axes[0, 1].set_title('Confidence Distribution Histogram', fontweight='bold', fontsize=20, pad=15)
        axes[0, 1].legend(fontsize=14)
        axes[0, 1].grid(True, alpha=0.3, linewidth=0.5)
        axes[0, 1].tick_params(labelsize=14)

        # 3. Confidence vs accuracy relationship
        correct_mask = (y_pred == y_true)
        confidence_correct = confidences[correct_mask]
        confidence_wrong = confidences[~correct_mask]

        axes[1, 0].hist(confidence_correct, bins=20, alpha=0.5, color='green',
                        label='Correct Predictions', edgecolor='black')
        axes[1, 0].hist(confidence_wrong, bins=20, alpha=0.5, color='red',
                        label='Wrong Predictions', edgecolor='black')
        axes[1, 0].set_xlabel('Confidence', fontweight='bold', fontsize=18)
        axes[1, 0].set_ylabel('Sample Count', fontweight='bold', fontsize=18)
        axes[1, 0].set_title('Confidence vs Prediction Correctness', fontweight='bold', fontsize=20, pad=15)
        axes[1, 0].legend(fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, linewidth=0.5)
        axes[1, 0].tick_params(labelsize=14)

        # 4. Per-class average confidence
        avg_confidence_per_class = []
        for i in range(n_classes):
            mask = y_true == i
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                avg_confidence_per_class.append(avg_conf)
            else:
                avg_confidence_per_class.append(0)

        bars = axes[1, 1].bar(range(n_classes), avg_confidence_per_class,
                              color=self.colors, edgecolor='black', linewidth=1.8)
        axes[1, 1].set_xlabel('Fault Class', fontweight='bold', fontsize=18)
        axes[1, 1].set_ylabel('Average Confidence', fontweight='bold', fontsize=18)
        axes[1, 1].set_title('Per-Class Average Diagnosis Confidence', fontweight='bold', fontsize=20, pad=15)
        axes[1, 1].set_xticks(range(n_classes))
        axes[1, 1].set_xticklabels(self.class_names, rotation=45, fontsize=14, ha='right')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5,
                           label='High Confidence Threshold (80%)')
        axes[1, 1].legend(fontsize=14)
        axes[1, 1].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[1, 1].tick_params(labelsize=14)

        # Add values on bars with larger font
        for bar, value in zip(bars, avg_confidence_per_class):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.suptitle('MC-MMamba Diagnosis Confidence Analysis',
                     fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'confidence_heatmap_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Confidence heatmap saved to: {save_path}")

        # Compute quantitative metrics
        metrics = {
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(np.std(confidences)),
            'confidence_correct_mean': float(np.mean(confidence_correct) if len(confidence_correct) > 0 else 0),
            'confidence_wrong_mean': float(np.mean(confidence_wrong) if len(confidence_wrong) > 0 else 0),
            'high_confidence_rate': float(np.sum(confidences >= 0.8) / len(confidences)),
            'confidence_per_class': [float(x) for x in avg_confidence_per_class]
        }

        return metrics

    def create_ablation_study(self, model, dataloader, save_path=None):
        """Create ablation study visualization with larger fonts - Enhanced with efficiency metrics"""
        model.eval()
        device = self.config.device

        # Define model variants for ablation study
        configs = [
            ('MC-MMamba (Full)', 'full', MC_MMamba(self.config)),
            ('MC-Net Only', 'mc_only', MCNetOnly(self.config)),
            ('Mamba Only', 'mamba_only', MambaOnly(self.config)),
            ('Vibration Only', 'vib_only', VibrationOnly(self.config)),
            ('Acoustic Only', 'audio_only', AcousticOnly(self.config))
        ]

        # Move models to device
        for _, _, model_variant in configs:
            model_variant.to(device)
            model_variant.eval()

        results = {}
        efficiency_metrics = {}

        with torch.no_grad():
            all_preds = {name: [] for name, _, _ in configs}
            all_labels = []

            # Warm-up run
            print("Warming up for inference timing...")
            warmup_data = next(iter(dataloader))
            vib_warmup, audio_warmup, _ = warmup_data
            vib_warmup = vib_warmup.to(device)[:2]
            audio_warmup = audio_warmup.to(device)[:2]

            for name, _, model_variant in configs:
                _ = model_variant(vib_warmup, audio_warmup)

            # Main evaluation loop
            for vib, audio, labels in tqdm(dataloader, desc='Ablation Study'):
                vib = vib.to(device)
                audio = audio.to(device)
                labels = labels.to(device)

                all_labels.extend(labels.cpu().numpy())

                # Test each variant
                for name, _, model_variant in configs:
                    # Time inference
                    start_time = time.time()
                    outputs = model_variant(vib, audio)
                    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    preds = torch.argmax(logits, dim=1)

                    all_preds[name].extend(preds.cpu().numpy())

                    # Store efficiency metrics
                    if name not in efficiency_metrics:
                        efficiency_metrics[name] = {
                            'inference_times': [],
                            'params': sum(p.numel() for p in model_variant.parameters()),
                            'trainable_params': sum(p.numel() for p in model_variant.parameters() if p.requires_grad)
                        }
                    efficiency_metrics[name]['inference_times'].append(inference_time / len(vib))  # Per sample time

        # Compute accuracy
        y_true = np.array(all_labels)
        accuracies = {}
        for name, _, _ in configs:
            y_pred = np.array(all_preds[name])
            accuracies[name] = accuracy_score(y_true, y_pred)

        # Compute efficiency metrics
        for name in efficiency_metrics:
            times = efficiency_metrics[name]['inference_times']
            efficiency_metrics[name]['avg_inference_time'] = np.mean(times)
            efficiency_metrics[name]['std_inference_time'] = np.std(times)
            efficiency_metrics[name]['params_million'] = efficiency_metrics[name]['params'] / 1e6
            efficiency_metrics[name]['trainable_params_million'] = efficiency_metrics[name]['trainable_params'] / 1e6

            # Compute FLOPs estimation (simplified)
            efficiency_metrics[name]['flops_estimate'] = efficiency_metrics[name]['params'] * 2  # Rough estimation

        # Create comprehensive visualization with larger figure size
        fig, axes = plt.subplots(2, 2, figsize=(22, 18), dpi=self.config.visualization_dpi)

        # 1. Accuracy comparison (top-left)
        config_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())

        bars1 = axes[0, 0].bar(range(len(config_names)), accuracy_values,
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                               edgecolor='black', linewidth=1.8)
        axes[0, 0].set_xlabel('Model Configuration', fontweight='bold', fontsize=18)
        axes[0, 0].set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        axes[0, 0].set_title('Ablation Study - Accuracy Comparison', fontweight='bold', fontsize=20, pad=15)
        axes[0, 0].set_xticks(range(len(config_names)))
        axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right', fontsize=14)
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[0, 0].tick_params(labelsize=14)

        # Add values on bars with larger font
        for bar, value in zip(bars1, accuracy_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 2. Parameter count comparison (top-right)
        param_counts = [efficiency_metrics[name]['params_million'] for name in config_names]
        trainable_params = [efficiency_metrics[name]['trainable_params_million'] for name in config_names]

        x = np.arange(len(config_names))
        width = 0.35

        bars2a = axes[0, 1].bar(x - width/2, param_counts, width, label='Total Parameters',
                                color='#1f77b4', edgecolor='black', linewidth=1.5)
        bars2b = axes[0, 1].bar(x + width/2, trainable_params, width, label='Trainable Parameters',
                                color='#ff7f0e', edgecolor='black', linewidth=1.5)

        axes[0, 1].set_xlabel('Model Configuration', fontweight='bold', fontsize=18)
        axes[0, 1].set_ylabel('Parameter Count (Millions)', fontweight='bold', fontsize=18)
        axes[0, 1].set_title('Model Complexity - Parameter Comparison', fontweight='bold', fontsize=20, pad=15)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(config_names, rotation=45, ha='right', fontsize=14)
        axes[0, 1].legend(fontsize=14)
        axes[0, 1].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[0, 1].tick_params(labelsize=14)

        # Add values on bars
        for i, (bar_total, bar_train) in enumerate(zip(bars2a, bars2b)):
            axes[0, 1].text(bar_total.get_x() + bar_total.get_width()/2., bar_total.get_height() + 0.1,
                            f'{param_counts[i]:.1f}M', ha='center', va='bottom', fontsize=11)
            axes[0, 1].text(bar_train.get_x() + bar_train.get_width()/2., bar_train.get_height() + 0.1,
                            f'{trainable_params[i]:.1f}M', ha='center', va='bottom', fontsize=11)

        # 3. Inference time comparison (bottom-left)
        inference_times = [efficiency_metrics[name]['avg_inference_time'] for name in config_names]
        inference_std = [efficiency_metrics[name]['std_inference_time'] for name in config_names]

        bars3 = axes[1, 0].bar(range(len(config_names)), inference_times,
                               yerr=inference_std, capsize=5,
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                               edgecolor='black', linewidth=1.8, alpha=0.8)
        axes[1, 0].set_xlabel('Model Configuration', fontweight='bold', fontsize=18)
        axes[1, 0].set_ylabel('Inference Time (ms per sample)', fontweight='bold', fontsize=18)
        axes[1, 0].set_title('Computational Efficiency - Inference Time', fontweight='bold', fontsize=20, pad=15)
        axes[1, 0].set_xticks(range(len(config_names)))
        axes[1, 0].set_xticklabels(config_names, rotation=45, ha='right', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[1, 0].tick_params(labelsize=14)

        # Add values on bars
        for bar, time_val in zip(bars3, inference_times):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(inference_times)*0.05,
                            f'{time_val:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 4. Efficiency-Accuracy trade-off (bottom-right)
        efficiency_scores = []
        for name in config_names:
            # Compute efficiency score (lower is better for time and params)
            time_score = 1.0 / (efficiency_metrics[name]['avg_inference_time'] + 1e-10)
            param_score = 1.0 / (efficiency_metrics[name]['params_million'] + 1e-10)
            efficiency_scores.append(time_score * param_score * 1000)  # Scaling factor

        # Normalize efficiency scores
        efficiency_scores = np.array(efficiency_scores)
        efficiency_scores = efficiency_scores / np.max(efficiency_scores)

        scatter = axes[1, 1].scatter(efficiency_scores, accuracy_values, s=300,
                                     c=range(len(config_names)), cmap='viridis', alpha=0.8,
                                     edgecolors='black', linewidth=2)

        axes[1, 1].set_xlabel('Efficiency Score (Normalized)', fontweight='bold', fontsize=18)
        axes[1, 1].set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        axes[1, 1].set_title('Efficiency-Accuracy Trade-off Analysis', fontweight='bold', fontsize=20, pad=15)
        axes[1, 1].grid(True, alpha=0.3, linewidth=0.5)
        axes[1, 1].tick_params(labelsize=14)

        # Add labels for each point
        for i, (eff, acc, name) in enumerate(zip(efficiency_scores, accuracy_values, config_names)):
            axes[1, 1].text(eff, acc, name, fontsize=12, ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Add Pareto frontier line
        pareto_indices = []
        for i in range(len(config_names)):
            dominated = False
            for j in range(len(config_names)):
                if (accuracy_values[j] >= accuracy_values[i] and
                        efficiency_scores[j] >= efficiency_scores[i] and
                        (accuracy_values[j] > accuracy_values[i] or efficiency_scores[j] > efficiency_scores[i])):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        if len(pareto_indices) > 1:
            pareto_indices = sorted(pareto_indices, key=lambda i: efficiency_scores[i])
            pareto_x = [efficiency_scores[i] for i in pareto_indices]
            pareto_y = [accuracy_values[i] for i in pareto_indices]
            axes[1, 1].plot(pareto_x, pareto_y, 'r--', linewidth=2.5, alpha=0.7, label='Pareto Frontier')
            axes[1, 1].legend(fontsize=14)

        plt.suptitle('MC-MMamba Ablation Study with Efficiency Analysis',
                     fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'ablation_study_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Ablation study results saved to: {save_path}")

        # Create detailed efficiency table
        print("\n" + "="*80)
        print("DETAILED EFFICIENCY METRICS:")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Params (M)':<12} {'Trainable (M)':<14} {'Inference (ms)':<15}")
        print("-" * 80)

        for name in config_names:
            acc = accuracies[name]
            params = efficiency_metrics[name]['params_million']
            trainable = efficiency_metrics[name]['trainable_params_million']
            inference = efficiency_metrics[name]['avg_inference_time']
            print(f"{name:<20} {acc:.4f}      {params:.2f}         {trainable:.2f}           {inference:.2f}")

        print("="*80)

        # Save efficiency metrics to CSV
        eff_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': accuracies[name],
                'Total_Params_M': efficiency_metrics[name]['params_million'],
                'Trainable_Params_M': efficiency_metrics[name]['trainable_params_million'],
                'Avg_Inference_Time_ms': efficiency_metrics[name]['avg_inference_time'],
                'Std_Inference_Time_ms': efficiency_metrics[name]['std_inference_time']
            }
            for name in config_names
        ])

        csv_path = os.path.join(self.config.tables_dir, f'efficiency_metrics_{self.config.start_time}.csv')
        eff_df.to_csv(csv_path, index=False)
        print(f"✓ Efficiency metrics saved to: {csv_path}")

        return accuracies, efficiency_metrics

    def create_calibration_curve(self, y_true, y_prob, save_path=None):
        """Create calibration curve - Evaluate prediction probability reliability with larger fonts"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=self.config.visualization_dpi)  # Increased figure size

        from sklearn.calibration import calibration_curve

        # 1. Overall calibration curve (using prediction correctness as binary label)
        confidences = np.max(y_prob, axis=1)
        correct_predictions = (np.argmax(y_prob, axis=1) == y_true).astype(int)

        if len(np.unique(correct_predictions)) > 1:
            try:
                prob_true, prob_pred = calibration_curve(correct_predictions, confidences,
                                                         n_bins=10, strategy='uniform')

                axes[0, 0].plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=2.5)
                axes[0, 0].plot(prob_pred, prob_true, "s-", linewidth=3.0, markersize=8,
                                label="MC-MMamba", color='#1f77b4')
                axes[0, 0].set_xlabel('Predicted Probability', fontweight='bold', fontsize=18)
                axes[0, 0].set_ylabel('Actual Frequency', fontweight='bold', fontsize=18)
                axes[0, 0].set_title('Calibration Curve (Overall, Based on Correctness)',
                                     fontweight='bold', fontsize=20, pad=15)
                axes[0, 0].legend(loc="lower right", fontsize=14)
                axes[0, 0].grid(True, alpha=0.3, linewidth=0.5)
                axes[0, 0].tick_params(labelsize=14)
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, 'Calibration curve cannot be computed\nInsufficient data or uneven distribution',
                                ha='center', va='center', transform=axes[0, 0].transAxes,
                                fontsize=14)
                axes[0, 0].set_title('Calibration Curve (Cannot Compute)', fontweight='bold', fontsize=20, pad=15)
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data for calibration curve',
                            ha='center', va='center', transform=axes[0, 0].transAxes,
                            fontsize=14)
            axes[0, 0].set_title('Calibration Curve (Insufficient Data)', fontweight='bold', fontsize=20, pad=15)

        # 2. Confidence binning accuracy
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        bin_accuracies = []
        for i in range(10):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                bin_accuracies.append(accuracy)
            else:
                bin_accuracies.append(0)

        bars = axes[0, 1].bar(range(10), bin_accuracies, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence Bin', fontweight='bold', fontsize=18)
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        axes[0, 1].set_title('Binned Confidence Accuracy', fontweight='bold', fontsize=20, pad=15)
        axes[0, 1].set_xticks(range(10))
        axes[0, 1].set_xticklabels([f'{i*10}%-\n{(i+1)*10}%' for i in range(10)], rotation=45, fontsize=12)
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2.5)
        axes[0, 1].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[0, 1].tick_params(labelsize=14)

        # Add values on bars with larger font
        for i, (bar, acc) in enumerate(zip(bars, bin_accuracies)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., acc + 0.02,
                            f'{acc:.2f}', ha='center', va='bottom', fontsize=11)

        # 3. Per-class calibration curves (one-vs-rest approach)
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(5))

        for i in range(min(5, len(self.class_names))):
            if np.sum(y_true_bin[:, i]) > 0:
                try:
                    prob_true_class, prob_pred_class = calibration_curve(
                        y_true_bin[:, i], y_prob[:, i], n_bins=5, strategy='quantile')

                    axes[1, 0].plot(prob_pred_class, prob_true_class, "o-", linewidth=2.5,
                                    markersize=6, label=self.class_names[i])
                except:
                    continue

        axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=2.5)
        axes[1, 0].set_xlabel('Predicted Probability', fontweight='bold', fontsize=18)
        axes[1, 0].set_ylabel('Actual Frequency', fontweight='bold', fontsize=18)
        axes[1, 0].set_title('Per-Class Calibration Curves (One-vs-Rest)',
                             fontweight='bold', fontsize=20, pad=15)
        axes[1, 0].legend(loc="lower right", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, linewidth=0.5)
        axes[1, 0].tick_params(labelsize=14)

        # 4. Reliability histogram
        expected_calibration_errors = []
        for i in range(10):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_mean_prob = np.mean(confidences[mask])
                bin_accuracy = np.mean(correct_predictions[mask])
                ece = np.abs(bin_mean_prob - bin_accuracy)
                expected_calibration_errors.append(ece)
            else:
                expected_calibration_errors.append(0)

        bars_ece = axes[1, 1].bar(range(10), expected_calibration_errors, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Confidence Bin', fontweight='bold', fontsize=18)
        axes[1, 1].set_ylabel('Calibration Error', fontweight='bold', fontsize=18)
        axes[1, 1].set_title('Expected Calibration Error (ECE)', fontweight='bold', fontsize=20, pad=15)
        axes[1, 1].set_xticks(range(10))
        axes[1, 1].set_xticklabels([f'{i*10}%-\n{(i+1)*10}%' for i in range(10)], rotation=45, fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, linewidth=0.5, axis='y')
        axes[1, 1].tick_params(labelsize=14)

        # Add values on bars with larger font
        for i, (bar, ece) in enumerate(zip(bars_ece, expected_calibration_errors)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., ece + 0.005,
                            f'{ece:.3f}', ha='center', va='bottom', fontsize=11)

        # Compute total ECE
        total_ece = np.sum(np.array(expected_calibration_errors) *
                           np.array([np.sum(bin_indices == i) for i in range(10)]) / len(y_true))

        axes[1, 1].text(0.95, 0.95, f'Total ECE: {total_ece:.4f}',
                        transform=axes[1, 1].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=14)

        plt.suptitle('Diagnosis Probability Reliability Assessment - Calibration Curve Analysis',
                     fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)

        if save_path is None:
            save_path = os.path.join(self.config.visualizations_dir,
                                     f'calibration_curve_{self.config.start_time}.png')

        plt.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight', facecolor='white')
        if self.config.save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        plt.show()
        plt.close(fig)
        print(f"✓ Calibration curve saved to: {save_path}")

        # Compute quantitative metrics
        metrics = {
            'expected_calibration_error': float(total_ece),
            'bin_accuracies': [float(x) for x in bin_accuracies],
            'calibration_errors': [float(x) for x in expected_calibration_errors]
        }

        return metrics

    def create_comprehensive_explainability_report(self, model, dataloader, y_true, y_pred,
                                                   y_prob, save_path=None):
        """Create comprehensive explainability report"""

        print("\n" + "="*80)
        print("Generating Comprehensive Explainability Report...")
        print("="*80)

        # Create report directory
        report_dir = os.path.join(self.config.results_dir, f"explainability_report_{self.config.start_time}")
        os.makedirs(report_dir, exist_ok=True)

        # 1. Grad-CAM heatmaps
        print("1. Generating Grad-CAM heatmaps...")
        grad_cam_path = os.path.join(report_dir, "grad_cam_heatmaps.png")
        try:
            self.create_grad_cam_heatmaps(model, dataloader, num_samples=5, save_path=grad_cam_path)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")

        # 2. Confidence analysis
        print("2. Generating confidence analysis...")
        confidence_path = os.path.join(report_dir, "confidence_analysis.png")
        try:
            confidence_metrics = self.create_confidence_heatmap(y_true, y_pred, y_prob, save_path=confidence_path)
        except Exception as e:
            print(f"Confidence analysis generation failed: {e}")
            confidence_metrics = {}

        # 3. Ablation study with efficiency analysis
        print("3. Generating ablation study with efficiency analysis...")
        ablation_path = os.path.join(report_dir, "ablation_study.png")
        try:
            ablation_results = self.create_ablation_study(model, dataloader, save_path=ablation_path)
            ablation_accuracies = ablation_results[0] if ablation_results else {}
            efficiency_metrics = ablation_results[1] if ablation_results else {}
        except Exception as e:
            print(f"Ablation study generation failed: {e}")
            ablation_accuracies = {}
            efficiency_metrics = {}

        # 4. Calibration curve
        print("4. Generating calibration curve...")
        calibration_path = os.path.join(report_dir, "calibration_curve.png")
        try:
            calibration_metrics = self.create_calibration_curve(y_true, y_prob, save_path=calibration_path)
        except Exception as e:
            print(f"Calibration curve generation failed: {e}")
            calibration_metrics = {}

        # 5. Generate comprehensive report
        print("5. Generating comprehensive report...")
        try:
            self._generate_explainability_summary(report_dir, confidence_metrics,
                                                  ablation_accuracies, efficiency_metrics,
                                                  calibration_metrics)
        except Exception as e:
            print(f"Comprehensive report generation failed: {e}")

        print(f"\n✅ Comprehensive Explainability Report Generation Complete!")
        print(f"Report Directory: {report_dir}")
        print("Contents:")
        print("  - Grad-CAM heatmaps (five-class fault visualization)")
        print("  - Confidence analysis charts")
        print("  - Ablation study with efficiency analysis (accuracy, parameters, inference time)")
        print("  - Calibration curve plots")
        print("  - Comprehensive assessment report (JSON)")

        return report_dir

    def _generate_explainability_summary(self, report_dir, confidence_metrics,
                                         ablation_accuracies, efficiency_metrics,
                                         calibration_metrics):
        """Generate explainability assessment summary"""

        summary = {
            "report_generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_config": {
                "fault_classes": self.class_names,
                "num_classes": len(self.class_names)
            },
            "confidence_analysis": confidence_metrics,
            "ablation_study": {
                "accuracies": ablation_accuracies,
                "efficiency_metrics": efficiency_metrics
            },
            "calibration_analysis": calibration_metrics,
            "overall_assessment": self._compute_overall_trustworthiness(
                confidence_metrics, ablation_accuracies, efficiency_metrics, calibration_metrics
            )
        }

        # Save JSON report
        json_path = os.path.join(report_dir, "explainability_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        # Generate text report
        txt_path = os.path.join(report_dir, "explainability_summary.txt")
        self._generate_text_report(txt_path, summary)

        print(f"✓ Explainability summary saved to: {json_path}")
        print(f"✓ Text report saved to: {txt_path}")

        return summary

    def _compute_overall_trustworthiness(self, confidence_metrics, ablation_accuracies,
                                         efficiency_metrics, calibration_metrics):
        """Compute overall trustworthiness score"""

        scores = {}

        # 1. Confidence score (0-100)
        if confidence_metrics:
            confidence_score = confidence_metrics.get('mean_confidence', 0.5) * 100
            high_confidence_rate = confidence_metrics.get('high_confidence_rate', 0) * 100
            scores['confidence_score'] = min(100, confidence_score * 0.7 + high_confidence_rate * 0.3)
        else:
            scores['confidence_score'] = 0

        # 2. Ablation study score (0-100)
        if ablation_accuracies and 'MC-MMamba (Full)' in ablation_accuracies:
            full_acc = ablation_accuracies['MC-MMamba (Full)']
            scores['ablation_score'] = min(100, full_acc * 100)
        else:
            scores['ablation_score'] = 0

        # 3. Efficiency score (0-100) - considering both accuracy and efficiency
        if ablation_accuracies and efficiency_metrics and 'MC-MMamba (Full)' in efficiency_metrics:
            full_params = efficiency_metrics['MC-MMamba (Full)']['params_million']
            full_time = efficiency_metrics['MC-MMamba (Full)']['avg_inference_time']
            full_acc = ablation_accuracies['MC-MMamba (Full)']

            # Compute efficiency relative to other models
            param_scores = []
            time_scores = []
            for name in ablation_accuracies:
                if name in efficiency_metrics:
                    param_ratio = full_params / max(1e-10, efficiency_metrics[name]['params_million'])
                    time_ratio = full_time / max(1e-10, efficiency_metrics[name]['avg_inference_time'])
                    acc_ratio = full_acc / max(1e-10, ablation_accuracies[name])

                    # Combined efficiency score
                    eff_score = (param_ratio * 0.4 + time_ratio * 0.4 + acc_ratio * 0.2) * 25  # Scale to 0-100
                    scores['efficiency_score'] = min(100, max(0, eff_score))
                    break
            else:
                scores['efficiency_score'] = 50  # Default
        else:
            scores['efficiency_score'] = 50

        # 4. Calibration score (0-100)
        if calibration_metrics:
            ece = calibration_metrics.get('expected_calibration_error', 0)
            calibration_score = max(0, 100 - ece * 1000)  # Smaller ECE is better
            scores['calibration_score'] = calibration_score
        else:
            scores['calibration_score'] = 0

        # 4. Overall trustworthiness score
        weights = {
            'confidence': 0.25,
            'ablation': 1.00,
            'efficiency': 0.20,
            'calibration': 0.30
        }

        overall_score = (
                scores['confidence_score'] * weights['confidence'] +
                scores['ablation_score'] * weights['ablation'] +
                scores['efficiency_score'] * weights['efficiency'] +
                scores['calibration_score'] * weights['calibration']
        )

        scores['overall_trustworthiness_score'] = overall_score

        # 5. Rating
        if overall_score >= 90:
            rating = "A+ (Highly Trustworthy & Efficient)"
        elif overall_score >= 80:
            rating = "A (Very Trustworthy & Efficient)"
        elif overall_score >= 70:
            rating = "B (Trustworthy & Balanced)"
        elif overall_score >= 60:
            rating = "C (Moderately Trustworthy)"
        else:
            rating = "D (Needs Improvement)"

        scores['trustworthiness_rating'] = rating

        return scores

    def _generate_text_report(self, filepath, summary):
        """Generate text format report"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MC-MMamba Explainability Analysis Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generation Time: {summary['report_generation_time']}\n")
            f.write(f"Number of Fault Classes: {summary['system_config']['num_classes']}\n")
            f.write(f"Fault Classes: {', '.join(summary['system_config']['fault_classes'])}\n\n")

            f.write("-"*80 + "\n")
            f.write("Overall Trustworthiness Assessment\n")
            f.write("-"*80 + "\n")

            overall = summary['overall_assessment']
            f.write(f"Overall Trustworthiness Score: {overall['overall_trustworthiness_score']:.1f}/100\n")
            f.write(f"Trustworthiness Rating: {overall['trustworthiness_rating']}\n\n")

            f.write(f"Confidence Score: {overall['confidence_score']:.1f}/100\n")
            f.write(f"Ablation Study Score: {overall['ablation_score']:.1f}/100\n")
            f.write(f"Efficiency Score: {overall['efficiency_score']:.1f}/100\n")
            f.write(f"Calibration Score: {overall['calibration_score']:.1f}/100\n\n")

            f.write("-"*80 + "\n")
            f.write("Detailed Analysis Results\n")
            f.write("-"*80 + "\n\n")

            # Confidence analysis
            if 'confidence_analysis' in summary:
                f.write("1. Confidence Analysis:\n")
                conf = summary['confidence_analysis']
                f.write(f"   Average Confidence: {conf.get('mean_confidence', 0):.3f}\n")
                f.write(f"   High Confidence Sample Ratio (≥80%): {conf.get('high_confidence_rate', 0):.2%}\n")
                f.write(f"   Average Confidence (Correct): {conf.get('confidence_correct_mean', 0):.3f}\n")
                f.write(f"   Average Confidence (Wrong): {conf.get('confidence_wrong_mean', 0):.3f}\n\n")

            # Ablation study and efficiency
            if 'ablation_study' in summary:
                f.write("2. Ablation Study & Efficiency Analysis:\n")
                abl = summary['ablation_study']
                if 'accuracies' in abl:
                    f.write("   Model Accuracies:\n")
                    for name, acc in abl['accuracies'].items():
                        f.write(f"     {name}: {acc:.4f}\n")

                if 'efficiency_metrics' in abl and 'MC-MMamba (Full)' in abl['efficiency_metrics']:
                    f.write("\n   Efficiency Metrics (Full Model):\n")
                    full_eff = abl['efficiency_metrics']['MC-MMamba (Full)']
                    f.write(f"     Total Parameters: {full_eff.get('params_million', 0):.2f}M\n")
                    f.write(f"     Trainable Parameters: {full_eff.get('trainable_params_million', 0):.2f}M\n")
                    f.write(f"     Average Inference Time: {full_eff.get('avg_inference_time', 0):.2f} ms/sample\n")
                f.write("\n")

            # Calibration analysis
            if 'calibration_analysis' in summary:
                f.write("3. Calibration Analysis:\n")
                cal = summary['calibration_analysis']
                f.write(f"   Expected Calibration Error (ECE): {cal.get('expected_calibration_error', 0):.4f}\n\n")

            f.write("="*80 + "\n")
            f.write("Diagnosis and Improvement Recommendations\n")
            f.write("="*80 + "\n\n")

            # Provide recommendations based on score
            overall_score = overall['overall_trustworthiness_score']

            if overall_score >= 85:
                f.write("✅ DIAGNOSIS: Model exhibits high trustworthiness and efficiency, suitable for deployment.\n")
                f.write("💡 RECOMMENDATION: Excellent performance, consider optimization for edge deployment.\n")
            elif overall_score >= 70:
                f.write("⚠️ DIAGNOSIS: Model trustworthiness is good with reasonable efficiency.\n")
                f.write("💡 RECOMMENDATION: Consider model pruning or quantization for better efficiency.\n")
            elif overall_score >= 60:
                f.write("⚠️ DIAGNOSIS: Model trustworthiness is moderate, efficiency could be improved.\n")
                f.write("💡 RECOMMENDATION: Optimize architecture, consider knowledge distillation.\n")
            else:
                f.write("❌ DIAGNOSIS: Model trustworthiness or efficiency is low.\n")
                f.write("💡 RECOMMENDATION: Redesign architecture, focus on efficiency-accuracy trade-off.\n")

            f.write("\n")
            f.write("="*80 + "\n")
            f.write("Report End\n")
            f.write("="*80 + "\n")

# ==================== 12. MAIN FUNCTION - FIVE-CLASS FAULT DIAGNOSIS (ENHANCED EXPLAINABILITY) ====================
def main_five_classes_with_explainability():
    """Five-Class Fault Diagnosis Main Function (with Explainability Analysis)"""

    print("="*80)
    print("MC-MMamba: Five-Class Fault Diagnosis System (Enhanced Explainability Version)")
    print("Fault Classes: cage, inner, normal, outer, roller")
    print("="*80)

    # 1. Initialize configuration
    print("\n1. Initializing system configuration...")
    config = SystemConfig()
    config.save('five_classes_config_explainable.json')

    print(f"Device: {config.device}")
    print(f"Number of Fault Classes: {config.num_classes}")
    print(f"Fault Classes: {SystemConfig.FAULT_CLASSES}")

    # 2. Load data
    print("\n2. Loading data files...")

    # Search data files
    all_mat = sorted(glob.glob('./data/**/*.mat', recursive=True))
    all_wav = sorted(glob.glob('./data/**/*.wav', recursive=True))

    print(f"Found {len(all_mat)} vibration files, {len(all_wav)} acoustic files")

    if len(all_mat) == 0 or len(all_wav) == 0:
        print("Error: No data files found!")
        print("Please place data files in ./data/ directory, organized by class:")
        for cls in SystemConfig.FAULT_CLASSES:
            print(f"  ./data/{cls}/ - store vibration (.mat) and acoustic (.wav) files for {cls} class")
        return

    # 3. Create five-class dataset
    print("\n3. Creating five-class fault dataset...")

    # Split data
    min_len = min(len(all_mat), len(all_wav))
    indices = np.arange(min_len)

    # First split into train+val and test sets
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Then split train+val into train and validation sets
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,  # 0.125 * 0.8 = 0.1
        random_state=42,
        shuffle=True
    )

    # Create datasets
    train_dataset = FiveClassDataset(
        [all_mat[i] for i in train_idx],
        [all_wav[i] for i in train_idx],
        config,
        is_train=True,
        augment=True
    )

    val_dataset = FiveClassDataset(
        [all_mat[i] for i in val_idx],
        [all_wav[i] for i in val_idx],
        config,
        is_train=False,
        augment=False
    )

    test_dataset = FiveClassDataset(
        [all_mat[i] for i in test_idx],
        [all_wav[i] for i in test_idx],
        config,
        is_train=False,
        augment=False
    )

    print(f"\nDataset Statistics:")
    print(f"  Training Set: {len(train_dataset)} samples")
    print(f"  Validation Set: {len(val_dataset)} samples")
    print(f"  Test Set: {len(test_dataset)} samples")

    # 4. Create data loaders
    print("\n4. Creating data loaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 5. Create model
    print("\n5. Creating MC-MMamba five-class fault diagnosis model...")
    model = MC_MMamba(config)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024 ** 2):.2f} MB")

    # 6. Train model (optional)
    train_model = True  # Can be set as needed

    if train_model:
        print("\n6. Starting five-class fault diagnosis model training...")
        trainer = FiveClassTrainer(model, config, train_dataset.class_weights)

        try:
            best_accuracy = trainer.train(train_loader, val_loader)
            print(f"\n✅ Training completed! Best validation accuracy: {best_accuracy:.2f}%")

            # Load best model
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                trainer.load_checkpoint(best_model_path)
                print(f"✓ Best model loaded: {best_model_path}")
        except Exception as e:
            print(f"✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return

    # 7. Test set evaluation
    print("\n7. Test set evaluation...")

    # Create evaluators
    evaluator = EnhancedFiveClassEvaluator(config)
    visualizer = FiveClassVisualizer(config)

    # Collect test set predictions
    test_features = []
    test_probs = []
    test_preds = []
    test_labels_list = []

    with torch.no_grad():
        for vib, audio, labels in tqdm(test_loader, desc='Test Set Evaluation'):
            vib = vib.to(config.device)
            audio = audio.to(config.device)

            outputs = model(vib, audio)
            features = outputs['fused_features']
            logits = outputs['logits']

            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            test_features.append(features.cpu().numpy())
            test_probs.append(probs.cpu().numpy())
            test_preds.append(preds.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    # Compute test set metrics (with trustworthiness)
    test_metrics = evaluator.compute_all_metrics_with_trustworthiness(test_labels, test_preds, test_probs)
    evaluator.print_detailed_report(test_labels, test_preds)

    print(f"\n📊 Trustworthiness Metrics:")
    print(f"   Average Confidence: {test_metrics.get('mean_confidence', 0):.4f}")
    print(f"   High Confidence Ratio (≥80%): {test_metrics.get('high_confidence_rate_80', 0):.2%}")
    print(f"   Expected Calibration Error: {test_metrics.get('expected_calibration_error', 0):.4f}")
    print(f"   Average Prediction Entropy: {test_metrics.get('mean_entropy', 0):.4f}")
    print(f"   Overall Trustworthiness Score: {test_metrics.get('overall_trustworthiness_score', 0):.1f}/100")

    # 8. Explainability analysis
    print("\n8. Starting explainability analysis...")

    explainability_analyzer = ExplainabilityAnalyzer(config)

    # Generate comprehensive explainability report
    report_dir = explainability_analyzer.create_comprehensive_explainability_report(
        model, test_loader, test_labels, test_preds, test_probs
    )

    # 9. Generate final summary
    print("\n" + "="*80)
    print("MC-MMamba Five-Class Fault Diagnosis System (Enhanced Explainability) Complete!")
    print("="*80)

    print(f"\n📊 DIAGNOSIS PERFORMANCE SUMMARY:")
    print(f"   Test Set Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Set F1-Score: {test_metrics['f1']:.4f}")
    print(f"   Test Set Mean AUC: {test_metrics.get('mean_auc', 0):.4f}")

    print(f"\n🔬 EXPLAINABILITY ANALYSIS SUMMARY:")
    print(f"   Report Directory: {report_dir}")
    print(f"   Analysis Included:")
    print(f"     - Grad-CAM heatmaps (visualize attention for five fault classes)")
    print(f"     - Confidence analysis (assess diagnosis trustworthiness)")
    print(f"     - Ablation study with efficiency analysis (accuracy, parameters, inference time)")
    print(f"     - Calibration curves (probability reliability assessment)")

    print(f"\n📈 IMPROVEMENT RECOMMENDATIONS:")
    print(f"   1. Focus on low-confidence samples, consider data augmentation")
    print(f"   2. Analyze ablation study results, optimize model architecture")
    print(f"   3. Use calibration techniques to improve probability reliability")
    print(f"   4. Consider model pruning or quantization for deployment efficiency")

    print("\n✅ All analyses completed, explainability report generated!")
    print("="*80)

# ==================== 13. QUICK TEST FUNCTION ====================
def quick_test_five_classes():
    """Five-class fault quick test"""
    print("Five-Class Fault Diagnosis Model Quick Test...")

    config = SystemConfig()

    # Test model creation
    try:
        model = MC_MMamba(config)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"✓ Five-class fault model created successfully")
        print(f"✓ Fault Classes: {SystemConfig.FAULT_CLASSES}")
        print(f"✓ Output Classes: {config.num_classes}")
        print(f"✓ Total Parameters: {total_params:,}")

        # Test forward pass
        model = model.to(config.device)
        batch_size = 2
        vib_signal = torch.randn(batch_size, config.signal_length).to(config.device)
        acoustic_signal = torch.randn(batch_size, config.signal_length).to(config.device)

        with torch.no_grad():
            outputs = model(vib_signal, acoustic_signal)

        print(f"✓ Input Shape: Vibration={vib_signal.shape}, Acoustic={acoustic_signal.shape}")
        print(f"✓ Output Shape: logits={outputs['logits'].shape}")
        print(f"✓ Output Classes: {outputs['logits'].shape[1]} (should be 5)")

        # Test dataset creation
        print("\nTesting dataset creation...")
        # Create mock data paths
        test_mat_paths = ['test1.mat', 'test2.mat', 'test3.mat']
        test_wav_paths = ['test1.wav', 'test2.wav', 'test3.wav']

        dataset = FiveClassDataset(test_mat_paths, test_wav_paths, config)
        print(f"✓ Five-class dataset created successfully")
        print(f"✓ Class Names: {dataset.class_names}")

        # Test visualizer
        visualizer = FiveClassVisualizer(config)
        print(f"✓ Five-class visualizer created successfully")

        # Test evaluator
        evaluator = FiveClassEvaluator(config)
        print(f"✓ Five-class evaluator created successfully")

        # Test enhanced evaluator
        enhanced_evaluator = EnhancedFiveClassEvaluator(config)
        print(f"✓ Enhanced evaluator created successfully")

        print("\n✅ All five-class fault diagnosis system components tested successfully!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test_explainability():
    """Quick test for explainability module"""

    print("Explainability Analysis Module Quick Test...")

    config = SystemConfig()

    try:
        # Test model creation
        model = MC_MMamba(config)
        print(f"✓ Five-class fault model created successfully")

        # Test dataset creation
        print("\nTesting dataset creation...")
        test_mat_paths = ['test1.mat', 'test2.mat', 'test3.mat']
        test_wav_paths = ['test1.wav', 'test2.wav', 'test3.wav']

        dataset = FiveClassDataset(test_mat_paths, test_wav_paths, config)
        print(f"✓ Five-class dataset created successfully")

        # Test explainability analyzer
        analyzer = ExplainabilityAnalyzer(config)
        print(f"✓ Explainability analyzer created successfully")

        # Test enhanced evaluator
        evaluator = EnhancedFiveClassEvaluator(config)
        print(f"✓ Enhanced evaluator created successfully")

        # Generate mock data for testing
        print("\nGenerating mock data for testing...")
        n_samples = 100
        y_true = np.random.randint(0, 5, n_samples)
        y_pred = y_true.copy()
        y_pred[10:20] = (y_pred[10:20] + 1) % 5  # Create some errors

        y_prob = np.random.rand(n_samples, 5)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        # Test trustworthiness metrics computation
        trust_metrics = evaluator.compute_trustworthiness_metrics(y_true, y_pred, y_prob)
        print(f"✓ Trustworthiness metrics computed successfully")
        print(f"  Average Confidence: {trust_metrics.get('mean_confidence', 0):.3f}")
        print(f"  High Confidence Ratio: {trust_metrics.get('high_confidence_rate_80', 0):.2%}")
        print(f"  Expected Calibration Error: {trust_metrics.get('expected_calibration_error', 0):.4f}")

        # Test comprehensive metrics computation
        all_metrics = evaluator.compute_all_metrics_with_trustworthiness(y_true, y_pred, y_prob)
        print(f"✓ Comprehensive metrics computed successfully")
        print(f"  Overall Trustworthiness Score: {all_metrics.get('overall_trustworthiness_score', 0):.1f}/100")

        print("\n✅ All explainability module components tested successfully!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 14. PROGRAM ENTRY POINT ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MC-MMamba Five-Class Fault Diagnosis System (Enhanced Explainability)')
    parser.add_argument('--mode', type=str, default='train_explain',
                        choices=['train', 'train_explain', 'test', 'quick_test',
                                 'explain_test', 'demo'],
                        help='Run mode')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory path')

    args = parser.parse_args()

    if args.mode == 'quick_test':
        quick_test_five_classes()
    elif args.mode == 'explain_test':
        quick_test_explainability()
    elif args.mode == 'test':
        print("Test mode...")
        quick_test_five_classes()
    elif args.mode == 'demo':
        print("Demo mode...")
        quick_test_explainability()
    elif args.mode == 'train_explain':
        print("MC-MMamba Five-Class Fault Diagnosis System - Training Mode (with Explainability Analysis)")
        print("-" * 60)
        print(f"Fault Classes: {SystemConfig.FAULT_CLASSES}")

        if not os.path.exists(args.data_dir):
            print(f"Warning: Data directory {args.data_dir} does not exist")
            print("Please organize data in the following structure:")
            for cls in SystemConfig.FAULT_CLASSES:
                print(f"  {args.data_dir}/{cls}/ - store vibration (.mat) and acoustic (.wav) files for {cls} class")
            print("\nCreate sample directory structure? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                os.makedirs(args.data_dir, exist_ok=True)
                for cls in SystemConfig.FAULT_CLASSES:
                    os.makedirs(os.path.join(args.data_dir, cls), exist_ok=True)
                print("Sample directory structure created, please place data files in corresponding directories")
                print(f"Data directory: {args.data_dir}")
            else:
                print("Please create data directory and place data files first")
                exit(1)

        try:
            main_five_classes_with_explainability()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"\nProgram runtime error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("MC-MMamba Five-Class Fault Diagnosis System - Standard Training Mode")
        print("-" * 60)
        try:
            # Here you could call the standard version main_five_classes()
            print("Standard training mode not implemented, please use --mode=train_explain")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"\nProgram runtime error: {e}")
            import traceback
            traceback.print_exc()