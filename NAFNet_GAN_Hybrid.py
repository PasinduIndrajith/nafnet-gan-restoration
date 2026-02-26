
Supports: Gaussian Blur, White Noise, Blocking Artifacts
Run on Kaggle with GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# GENERATOR: MaskedNAFNet 


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class MaskedNAFNet(nn.Module):
    """
    NAFNet Generator - accepts image + mask (4 channels)
    This is the EXACT same architecture that achieved 30-33 dB PSNR standalone
    Now used as the Generator in our GAN framework
    """
    def __init__(self, img_channel=4, width=32, middle_blk_num=12,
                 enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, 3, 3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:, :3, :, :]  # Residual learning: output = input_rgb + learned_correction

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x



# DISCRIMINATOR: PatchGAN


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field)
    Input: degraded image (3ch) + output/target (3ch) = 6 channels
    NOTE: We use 6ch (not 7ch) because discriminator only sees the RGB images,
    not the mask. The mask is only for the generator.
    """
    def __init__(self, in_channels=6):
        super().__init__()

        def discriminator_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, normalize=False),  # 256->128
            discriminator_block(64, 128),                            # 128->64
            discriminator_block(128, 256),                           # 64->32
            discriminator_block(256, 512, stride=1),                 # 32->31
            nn.Conv2d(512, 1, 4, padding=1),                        # 31->30
        )

    def forward(self, img_input, img_output):
        """
        img_input: degraded RGB image (3 channels, extracted from 4ch input)
        img_output: restored or ground truth image (3 channels)
        """
        combined = torch.cat([img_input, img_output], dim=1)
        return self.model(combined)


=
# SSIM LOSS (Differentiable for training)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM Loss for training.
    Returns 1 - SSIM (so lower is better, suitable for minimization)
    Best for: structural artifacts like blur and blocking
    """
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels

        # Create gaussian kernel for SSIM calculation
        kernel_1d = self._gaussian_kernel_1d(window_size, 1.5)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', kernel)

    def _gaussian_kernel_1d(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()  # 1 - SSIM (so loss decreases as SSIM improves)



# COMBINED LOSS: L1 + SSIM + Adversarial


class NAFNetGANLoss:
    """
    Combined loss for NAFNet-GAN:
      Generator Loss = λ_L1 × L1 + λ_SSIM × (1-SSIM) + λ_GAN × GAN_loss
    
    Weights chosen carefully:
      - L1 (λ=100): Primary reconstruction loss (pixel accuracy)
      - SSIM (λ=50): Structural preservation (edges, textures)
      - GAN (λ=1): Adversarial sharpness (small weight to not overpower)
    
    L1 and SSIM work TOGETHER (not competing) because both want
    the output to match the ground truth. GAN just adds sharpness.
    """
    def __init__(self, lambda_l1=100.0, lambda_ssim=100.0, lambda_gan=0.5):
        self.criterion_gan = nn.MSELoss()  # LSGAN (stable training)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ssim = SSIMLoss(window_size=11, channels=3).to(device)

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_gan = lambda_gan

    def generator_loss(self, pred_fake, fake_output, real_target):
        """Generator wants: match ground truth (L1 + SSIM) AND fool D (GAN)"""
        real_label = torch.ones_like(pred_fake)

        loss_gan = self.criterion_gan(pred_fake, real_label)
        loss_l1 = self.criterion_l1(fake_output, real_target)
        loss_ssim = self.criterion_ssim(fake_output, real_target)

        total = (self.lambda_gan * loss_gan +
                 self.lambda_l1 * loss_l1 +
                 self.lambda_ssim * loss_ssim)

        return total, loss_gan, loss_l1, loss_ssim

    def discriminator_loss(self, pred_real, pred_fake):
        """Discriminator wants: correctly classify real vs fake"""
        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)
        loss_real = self.criterion_gan(pred_real, real_label)
        loss_fake = self.criterion_gan(pred_fake, fake_label)
        return (loss_real + loss_fake) * 0.5



# METRICS: PSNR and SSIM


def calculate_psnr(pred, target):
    """PSNR between tensors in [0, 1] range"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())

def calculate_ssim_metric(pred, target, window_size=11):
    """SSIM metric (not loss) for evaluation"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


# DATASET 


class MaskedRestorationDataset(Dataset):
    """Compatible with existing pipeline - normalizes to [0, 1]
    Now with data augmentation: random flips + random 90° rotations
    """
    def __init__(self, degraded_dir, mask_dir, sharp_dir, size=256, augment=True):
        self.degraded_paths = sorted(glob.glob(f'{degraded_dir}/*.png'))
        self.mask_paths = sorted(glob.glob(f'{mask_dir}/*.png'))
        self.sharp_dir = sharp_dir
        self.size = size
        self.augment = augment
        print(f"  Found {len(self.degraded_paths)} degraded images")
        print(f"  Found {len(self.mask_paths)} masks")
        print(f"  Data augmentation: {'ON ✅' if augment else 'OFF'}")

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        degraded = cv2.imread(self.degraded_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        degraded_name = Path(self.degraded_paths[idx]).stem
        sharp_name = degraded_name.split('_var')[0]
        sharp_path = f'{self.sharp_dir}/{sharp_name}.bmp'
        sharp = cv2.imread(sharp_path)

        degraded = cv2.resize(degraded, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        sharp = cv2.resize(sharp, (self.size, self.size))

        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        # ===== Data Augmentation =====
        # Applied IDENTICALLY to degraded, mask, and sharp
        if self.augment:
            # Random horizontal flip (50% chance)
            if np.random.random() > 0.5:
                degraded = np.flip(degraded, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                sharp = np.flip(sharp, axis=1).copy()
            
            # Random vertical flip (50% chance)
            if np.random.random() > 0.5:
                degraded = np.flip(degraded, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
                sharp = np.flip(sharp, axis=0).copy()
            
            # Random 90° rotation (0, 90, 180, or 270 degrees)
            k = np.random.randint(0, 4)  # 0=0°, 1=90°, 2=180°, 3=270°
            if k > 0:
                degraded = np.rot90(degraded, k).copy()
                mask = np.rot90(mask, k).copy()
                sharp = np.rot90(sharp, k).copy()

        # Normalize to [0, 1] (same as original NAFNet, NOT [-1, 1])
        degraded = degraded.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        sharp = sharp.astype(np.float32) / 255.0

        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        sharp = torch.from_numpy(sharp).permute(2, 0, 1)

        input_tensor = torch.cat([degraded, mask], dim=0)  # 4 channels
        return input_tensor, sharp


# CONFIGURATION


ARTIFACT_TYPE = 'blur'  # Options: 'blur', 'noise', 'blocking'

if ARTIFACT_TYPE == 'blur':
    DATASET_PATH = '/kaggle/input/datasets/wmpibweerasinghe/iqa-gblur'
    DEGRADED_DIR = '/kaggle/working/blur_mask_dataset/images'
    MASK_DIR = '/kaggle/working/blur_mask_dataset/masks'
    SHARP_DIR = f'{DATASET_PATH}/Original_New'
    CHECKPOINT_DIR = '/kaggle/working/nafnet_gan_blur_checkpoints'
    MODEL_NAME = 'NAFNet-GAN - Gaussian Blur Correction'

elif ARTIFACT_TYPE == 'noise':
    DATASET_PATH = '/kaggle/input/iqa-wn'
    DEGRADED_DIR = '/kaggle/working/noise_mask_dataset/images'
    MASK_DIR = '/kaggle/working/noise_mask_dataset/masks'
    SHARP_DIR = f'{DATASET_PATH}/Original_New'
    CHECKPOINT_DIR = '/kaggle/working/nafnet_gan_noise_checkpoints'
    MODEL_NAME = 'NAFNet-GAN - White Noise Correction'

elif ARTIFACT_TYPE == 'blocking':
    DATASET_PATH = '/kaggle/input/iqa-blocking'
    DEGRADED_DIR = '/kaggle/working/blocking_mask_dataset/images'
    MASK_DIR = '/kaggle/working/blocking_mask_dataset/masks'
    SHARP_DIR = f'{DATASET_PATH}/Original_New'
    CHECKPOINT_DIR = '/kaggle/working/nafnet_gan_blocking_checkpoints'
    MODEL_NAME = 'NAFNet-GAN - Blocking Artifact Correction'

# ===== Hyperparameters =====
IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 1000
LR_GENERATOR = 1e-4     # Same as standalone NAFNet
LR_DISCRIMINATOR = 1e-4  # Match generator LR
LAMBDA_L1 = 100.0        # Pixel accuracy
LAMBDA_SSIM = 100.0      # Structural quality - boosted (was 50, now equal to L1)
LAMBDA_GAN = 0.5         # Adversarial sharpness - reduced (was 1.0, prevents GAN from overpowering)
EMA_DECAY = 0.999        # EMA smoothing factor (keeps running average of generator weights)
WARMUP_EPOCHS = 10       # LR warmup for first 10 epochs

# Milestone epochs to save detailed output images
MILESTONE_EPOCHS = [10, 250, 500, 1000]

print(f"\n{'='*60}")
print(f"  {MODEL_NAME}")
print(f"  NAFNet Generator + PatchGAN Discriminator")
print(f"  Loss: L1 + SSIM + Adversarial")
print(f"{'='*60}")
print(f"  Artifact Type:   {ARTIFACT_TYPE}")
print(f"  Image Size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Batch Size:      {BATCH_SIZE}")
print(f"  Epochs:          {NUM_EPOCHS}")
print(f"  LR (G):          {LR_GENERATOR}")
print(f"  LR (D):          {LR_DISCRIMINATOR}")
print(f"  Lambda L1:       {LAMBDA_L1}")
print(f"  Lambda SSIM:     {LAMBDA_SSIM}")
print(f"  Lambda GAN:      {LAMBDA_GAN}")
print(f"{'='*60}\n")

# SETUP


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(f'{CHECKPOINT_DIR}/samples', exist_ok=True)

dataset = MaskedRestorationDataset(DEGRADED_DIR, MASK_DIR, SHARP_DIR, size=IMAGE_SIZE, augment=True)
print(f"Dataset size: {len(dataset)}")

# Validation set without augmentation
val_dataset_full = MaskedRestorationDataset(DEGRADED_DIR, MASK_DIR, SHARP_DIR, size=IMAGE_SIZE, augment=False)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
_, val_dataset = torch.utils.data.random_split(val_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {len(train_dataset)} (augmented), Val: {len(val_dataset)} (no augmentation)")

# Generator = NAFNet (same config as standalone version)
generator = MaskedNAFNet(
    img_channel=4,
    width=32,
    middle_blk_num=12,
    enc_blk_nums=[2, 2, 4, 8],
    dec_blk_nums=[2, 2, 2, 2]
).to(device)

# Discriminator = PatchGAN
discriminator = PatchGANDiscriminator(in_channels=6).to(device)

gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())
print(f"\nNAFNet Generator params:    {gen_params:,}")
print(f"PatchGAN Discriminator:     {disc_params:,}")
print(f"Total parameters:           {gen_params + disc_params:,}")

# Optimizers
optimizer_G = optim.AdamW(generator.parameters(), lr=LR_GENERATOR, weight_decay=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_DISCRIMINATOR, betas=(0.5, 0.999))

# LR Schedulers with Warmup
# Warmup: linearly increase LR from 1e-6 to LR_GENERATOR over WARMUP_EPOCHS
# Then: CosineAnnealing from LR_GENERATOR to 1e-6 for remaining epochs
def lr_lambda_warmup(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # Linear warmup: 0.1 → 1.0
    else:
        # Cosine annealing after warmup
        progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
        return max(1e-2, 0.5 * (1 + math.cos(math.pi * progress)))  # min 1% of LR

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda_warmup)
scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda_warmup)

print(f"\n📈 LR Schedule: Warmup ({WARMUP_EPOCHS} epochs) + Cosine Annealing")

# EMA (Exponential Moving Average) of Generator
# Keeps a smoothed copy of weights → reduces noise → better final model
ema_generator = copy.deepcopy(generator)
ema_generator.eval()
for p in ema_generator.parameters():
    p.requires_grad_(False)

def update_ema(ema_model, model, decay=0.999):
    """Update EMA weights: ema = decay * ema + (1 - decay) * current"""
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)

print(f"🔄 EMA enabled (decay={EMA_DECAY})")

# Loss
loss_fn = NAFNetGANLoss(
    lambda_l1=LAMBDA_L1,
    lambda_ssim=LAMBDA_SSIM,
    lambda_gan=LAMBDA_GAN
)



# VISUALIZATION


def save_sample_images(generator, val_loader, epoch, save_dir):
    """Save quick sample grid during regular training checkpoints"""
    generator.eval()
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        fake = generator(inputs)

        inputs_rgb = inputs[:, :3, :, :]
        masks = inputs[:, 3:4, :, :]

        n = min(4, inputs.shape[0])
        fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            img_in = inputs_rgb[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(np.clip(img_in, 0, 1))
            axes[i, 0].set_title('Degraded Input', fontsize=10)
            axes[i, 0].axis('off')

            mask_img = masks[i, 0].cpu().numpy()
            axes[i, 1].imshow(mask_img, cmap='gray')
            axes[i, 1].set_title('Detection Mask', fontsize=10)
            axes[i, 1].axis('off')

            img_fake = fake[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 2].imshow(np.clip(img_fake, 0, 1))
            axes[i, 2].set_title('NAFNet-GAN Output', fontsize=10)
            axes[i, 2].axis('off')

            img_gt = targets[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 3].imshow(np.clip(img_gt, 0, 1))
            axes[i, 3].set_title('Ground Truth', fontsize=10)
            axes[i, 3].axis('off')

        plt.suptitle(f'{MODEL_NAME} - Epoch {epoch+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/samples/epoch_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    generator.train()


def save_milestone_images(generator, val_loader, epoch, save_dir, label, psnr_val=None, ssim_val=None):
    """
    Save detailed output images at milestone epochs and best PSNR.
    Saves:
      1. A comparison grid (Degraded | Mask | NAFNet-GAN Output | Ground Truth)
      2. Individual restored images as standalone PNG files
    """
    generator.eval()
    milestone_dir = f'{save_dir}/milestone_{label}'
    os.makedirs(milestone_dir, exist_ok=True)

    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        fake = generator(inputs)

        inputs_rgb = inputs[:, :3, :, :]
        masks = inputs[:, 3:4, :, :]

        n = min(4, inputs.shape[0])

        # === 1. Save comparison grid ===
        fig, axes = plt.subplots(n, 4, figsize=(20, 5*n))
        if n == 1:
            axes = axes[np.newaxis, :]

        title_str = f'{MODEL_NAME} - {label}'
        if psnr_val is not None:
            title_str += f' | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}'

        for i in range(n):
            # Degraded input
            img_in = inputs_rgb[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(np.clip(img_in, 0, 1))
            axes[i, 0].set_title('Degraded Input', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            # Mask
            mask_img = masks[i, 0].cpu().numpy()
            axes[i, 1].imshow(mask_img, cmap='gray')
            axes[i, 1].set_title('Detection Mask', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            # NAFNet-GAN output
            img_fake = fake[i].cpu().permute(1, 2, 0).numpy()
            img_fake_clipped = np.clip(img_fake, 0, 1)
            axes[i, 2].imshow(img_fake_clipped)
            axes[i, 2].set_title('NAFNet-GAN Output', fontsize=12, fontweight='bold', color='green')
            axes[i, 2].axis('off')

            # Ground truth
            img_gt = targets[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 3].imshow(np.clip(img_gt, 0, 1))
            axes[i, 3].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')

            # === 2. Save individual restored images as PNG ===
            # Save restored output
            restored_uint8 = (img_fake_clipped * 255).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{milestone_dir}/restored_sample_{i+1}.png', restored_bgr)

            # Save degraded input for comparison
            degraded_uint8 = (np.clip(img_in, 0, 1) * 255).astype(np.uint8)
            degraded_bgr = cv2.cvtColor(degraded_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{milestone_dir}/degraded_sample_{i+1}.png', degraded_bgr)

            # Save ground truth for comparison
            gt_uint8 = (np.clip(img_gt, 0, 1) * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{milestone_dir}/groundtruth_sample_{i+1}.png', gt_bgr)

        plt.suptitle(title_str, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{milestone_dir}/comparison_grid.png', dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

    print(f"  📸 Milestone images saved to: {milestone_dir}/")
    print(f"     - comparison_grid.png (side-by-side comparison)")
    print(f"     - restored_sample_1..{n}.png (individual restored images)")
    print(f"     - degraded_sample_1..{n}.png (input images)")
    print(f"     - groundtruth_sample_1..{n}.png (ground truth images)")
    generator.train()

# TRAINING LOOP


g_losses = []
d_losses = []
val_psnrs = []
val_ssims = []
best_val_psnr = 0.0
best_epoch = 0

print(f"\n🚀 Starting NAFNet-GAN training for {ARTIFACT_TYPE} correction...\n")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    generator.train()
    discriminator.train()

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_l1_loss = 0.0
    epoch_ssim_loss = 0.0
    epoch_gan_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Extract RGB from 4-channel input for discriminator
        input_rgb = inputs[:, :3, :, :]

        # =====================
        # Train Discriminator
        # =====================
        optimizer_D.zero_grad()

        fake_output = generator(inputs)

        # Discriminator sees: [degraded_RGB + target] vs [degraded_RGB + fake]
        pred_real = discriminator(input_rgb, targets)
        pred_fake = discriminator(input_rgb, fake_output.detach())

        d_loss = loss_fn.discriminator_loss(pred_real, pred_fake)
        d_loss.backward()
        optimizer_D.step()

        # =====================
        # Train Generator
        # =====================
        optimizer_G.zero_grad()

        fake_output = generator(inputs)
        pred_fake = discriminator(input_rgb, fake_output)

        g_loss, gan_loss, l1_loss, ssim_loss = loss_fn.generator_loss(
            pred_fake, fake_output, targets
        )
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_l1_loss += l1_loss.item()
        epoch_ssim_loss += ssim_loss.item()
        epoch_gan_loss += gan_loss.item()

        # Update EMA weights after each batch
        update_ema(ema_generator, generator, decay=EMA_DECAY)

    epoch_g_loss /= len(train_loader)
    epoch_d_loss /= len(train_loader)
    epoch_l1_loss /= len(train_loader)
    epoch_ssim_loss /= len(train_loader)
    epoch_gan_loss /= len(train_loader)

    g_losses.append(epoch_g_loss)
    d_losses.append(epoch_d_loss)

    scheduler_G.step()
    scheduler_D.step()

    # =====================
    # Validation (every 10 epochs) — using EMA generator for better quality
    # =====================
    if (epoch + 1) % 10 == 0:
        ema_generator.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_val = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                fake = ema_generator(val_inputs)  # Use EMA model for validation

                for i in range(fake.shape[0]):
                    total_psnr += calculate_psnr(
                        torch.clamp(fake[i:i+1], 0, 1),
                        val_targets[i:i+1]
                    )
                    total_ssim += calculate_ssim_metric(
                        torch.clamp(fake[i:i+1], 0, 1),
                        val_targets[i:i+1]
                    )
                    num_val += 1

        avg_psnr = total_psnr / num_val
        avg_ssim = total_ssim / num_val
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        elapsed = (time.time() - start_time) / 60
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"G: {epoch_g_loss:.3f} (GAN: {epoch_gan_loss:.3f}, L1: {epoch_l1_loss:.4f}, SSIM: {epoch_ssim_loss:.3f}) | "
              f"D: {epoch_d_loss:.3f} | "
              f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}min")

        # Save milestone images ONLY at specific epochs (10, 250, 500)
        if (epoch + 1) in MILESTONE_EPOCHS:
            save_milestone_images(ema_generator, val_loader, epoch, CHECKPOINT_DIR,
                                 label=f'epoch_{epoch+1}', psnr_val=avg_psnr, ssim_val=avg_ssim)

        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            best_ssim = avg_ssim
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'generator_state_dict': ema_generator.state_dict(),  # Save EMA weights (smoother)
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'artifact_type': ARTIFACT_TYPE,
                'model': 'NAFNet-GAN + EMA',
            }, f'{CHECKPOINT_DIR}/nafnet_gan_best.pth')
            print(f"  💾 New best (EMA)! PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f} (epoch {epoch+1})")

    elif (epoch + 1) % 20 == 0:
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"G: {epoch_g_loss:.3f} | D: {epoch_d_loss:.3f} | Time: {elapsed:.1f}min")

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'artifact_type': ARTIFACT_TYPE,
        }, f'{CHECKPOINT_DIR}/nafnet_gan_epoch_{epoch+1}.pth')


# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = (time.time() - start_time) / 60
print(f"\n{'='*60}")
print(f"  Training Complete!")
print(f"{'='*60}")
print(f"  Model:     {MODEL_NAME}")
print(f"  Time:      {total_time:.1f} minutes")
print(f"  Best PSNR: {best_val_psnr:.2f} dB (at epoch {best_epoch})")
print(f"  Best SSIM: {best_ssim:.4f}")
print(f"{'='*60}")
print(f"\n  Comparison:")
print(f"  ┌─────────────────┬──────────┬─────────┐")
print(f"  │ Method          │ PSNR     │ SSIM    │")
print(f"  ├─────────────────┼──────────┼─────────┤")
print(f"  │ Classical       │ 23-27 dB │ ~0.85   │")
print(f"  │ Pix2Pix GAN     │ 27.40 dB │ 0.9150  │")
print(f"  │ NAFNet (alone)  │ 30-33 dB │ ~0.95   │")
print(f"  │ NAFNet-GAN      │ {best_val_psnr:.2f} dB │ {best_ssim:.4f} │")
print(f"  └─────────────────┴──────────┴─────────┘")
print(f"{'='*60}\n")

# ============================================================================
# LOAD BEST MODEL AND SAVE BEST PSNR IMAGES (only once, after training)
# ============================================================================

print("\n📸 Loading best EMA model and saving best PSNR output images...")
best_checkpoint = torch.load(f'{CHECKPOINT_DIR}/nafnet_gan_best.pth')
ema_generator.load_state_dict(best_checkpoint['generator_state_dict'])
print(f"   Loaded best EMA checkpoint from epoch {best_checkpoint['epoch']+1}")
print(f"   PSNR: {best_checkpoint['psnr']:.2f} dB | SSIM: {best_checkpoint['ssim']:.4f}")

save_milestone_images(ema_generator, val_loader, best_checkpoint['epoch'], CHECKPOINT_DIR,
                     label='best_psnr',
                     psnr_val=best_checkpoint['psnr'],
                     ssim_val=best_checkpoint['ssim'])


# ============================================================================
# PLOTS
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(g_losses, label='Generator', color='#2196F3', alpha=0.8)
axes[0].plot(d_losses, label='Discriminator', color='#FF5722', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title(f'{MODEL_NAME}\nTraining Losses')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

eval_epochs = list(range(10, NUM_EPOCHS + 1, 10))[:len(val_psnrs)]
axes[1].plot(eval_epochs, val_psnrs, 'o-', color='#4CAF50', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('Validation PSNR')
axes[1].grid(True, alpha=0.3)
if val_psnrs:
    axes[1].axhline(y=max(val_psnrs), color='r', linestyle='--', alpha=0.5,
                    label=f'NAFNet-GAN Best: {max(val_psnrs):.2f} dB')
    axes[1].axhline(y=27.40, color='orange', linestyle=':', alpha=0.5,
                    label='Pix2Pix: 27.40 dB')
    axes[1].legend()

axes[2].plot(eval_epochs, val_ssims, 'o-', color='#9C27B0', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('SSIM')
axes[2].set_title('Validation SSIM')
axes[2].grid(True, alpha=0.3)
if val_ssims:
    axes[2].axhline(y=max(val_ssims), color='r', linestyle='--', alpha=0.5,
                    label=f'NAFNet-GAN Best: {max(val_ssims):.4f}')
    axes[2].axhline(y=0.9150, color='orange', linestyle=':', alpha=0.5,
                    label='Pix2Pix: 0.9150')
    axes[2].legend()

plt.tight_layout()
plt.savefig(f'{CHECKPOINT_DIR}/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ All outputs saved to: {CHECKPOINT_DIR}/")
print(f"✅ Best model: {CHECKPOINT_DIR}/nafnet_gan_best.pth")
