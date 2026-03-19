import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
from monai.networks.nets import resnet18
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH  = "/scratch90/russ/GAclassifier2/OCT/output/binary_ga_resnet_v1/checkpoints/best-epoch=26-loss/val=0.3635.ckpt"
VOLUME_DIR = "/scratch90/russ/GAclassifier2/OCT/volumes"   # subfolders = volumes, PNGs inside = slices
OUTPUT_CSV = "/scratch90/russ/GAclassifier2/OCT/oct_predictions.csv"
THRESHOLD  = 0.5
IMAGE_SIZE = 256
NUM_FRAMES = 49
BATCH_SIZE = 12  # volumes per batch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── Attention modules (must match training definitions) ───────────────────────
class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


# ── Model ─────────────────────────────────────────────────────────────────────
class BinaryMIL_ResNet(nn.Module):
    def __init__(self, feature_dim=512, attention_dim=128, gated=True, dropout=False):
        super().__init__()
        feature_extractor = resnet18(spatial_dims=2, n_input_channels=1, num_classes=1)
        feature_extractor.fc = nn.Identity()
        self.feature_extractor = feature_extractor

        attn_cls = Attn_Net_Gated if gated else Attn_Net
        self.attention_net = attn_cls(L=feature_dim, D=attention_dim, dropout=dropout)
        self.classifier    = nn.Linear(feature_dim, 1)

    def forward(self, volume):
        """volume: (B, 1, H, W, D) → logits: (B,)"""
        B, C, H, W, D = volume.shape
        x     = volume.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        feats = self.feature_extractor(x).reshape(B, D, -1)
        A     = F.softmax(self.attention_net(feats), dim=1)
        agg   = torch.bmm(A.permute(0, 2, 1), feats).squeeze(1)
        return self.classifier(agg).squeeze(1)


def load_model(ckpt_path, device):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    # Lightning prefix is the module itself (no extra wrapper prefix for BinaryMIL_ResNet)
    # Keys look like 'feature_extractor.*', 'attention_net.*', 'classifier.*'
    # Strip any 'model.' prefix if present, otherwise use as-is
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v
    model = BinaryMIL_ResNet()
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


# ── Volume loader ─────────────────────────────────────────────────────────────
slice_transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
])


def load_volume(volume_dir, num_frames):
    """Load all PNGs in a folder as a volume tensor (1, H, W, D)."""
    slice_paths = sorted([
        os.path.join(volume_dir, f)
        for f in os.listdir(volume_dir)
        if f.lower().endswith('.png')
    ])
    slices = []
    for sp in slice_paths:
        img = Image.open(sp).convert('L')  # grayscale
        t   = v2.functional.to_image(img)  # (1, H, W) uint8
        t   = slice_transform(t)           # (1, H, W) float32
        slices.append(t)

    if not slices:
        return None

    volume = torch.stack(slices, dim=-1)   # (1, H, W, D)
    D = volume.shape[-1]
    if D < num_frames:
        volume = F.pad(volume, (0, num_frames - D))
    else:
        volume = volume[..., :num_frames]

    return volume  # (1, H, W, num_frames)


def get_volume_dirs(volume_root):
    dirs = []
    for name in sorted(os.listdir(volume_root)):
        path = os.path.join(volume_root, name)
        if os.path.isdir(path):
            dirs.append((name, path))
    return dirs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(CKPT_PATH, device)
    print(f"Model loaded from {CKPT_PATH}")

    volume_dirs = get_volume_dirs(VOLUME_DIR)
    print(f"Found {len(volume_dirs)} volume subfolders in {VOLUME_DIR}")

    rows = []
    for i in tqdm(range(0, len(volume_dirs), BATCH_SIZE), desc="Inferring", unit="batch"):
        batch_info    = volume_dirs[i:i + BATCH_SIZE]
        batch_volumes = []
        batch_names   = []
        batch_paths   = []

        for name, vdir in batch_info:
            vol = load_volume(vdir, NUM_FRAMES)
            if vol is None:
                print(f"  Skipping {vdir} — no PNGs found")
                continue
            batch_volumes.append(vol)
            batch_names.append(name)
            batch_paths.append(vdir)

        if not batch_volumes:
            continue

        volumes = torch.stack(batch_volumes).to(device)  # (B, 1, H, W, D)
        with torch.no_grad():
            logits = model(volumes)
            probs  = torch.sigmoid(logits).cpu().numpy()

        preds = (probs >= THRESHOLD).astype(int)
        for name, vdir, prob, pred in zip(batch_names, batch_paths, probs, preds):
            rows.append({
                'volume_name': name,
                'filepath':    vdir,
                'probability': round(float(prob), 4),
                'prediction':  int(pred),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} predictions to {OUTPUT_CSV}")
    print(f"  GA+: {df['prediction'].sum()}  |  GA-: {(df['prediction'] == 0).sum()}")


if __name__ == "__main__":
    main() 