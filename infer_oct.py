import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
from monai.networks.nets import resnet18
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="OCT MIL inference script")
    parser.add_argument("--ckpt_path",  type=str, required=True)
    parser.add_argument("--volume_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--image_size", type=int,   default=256)
    parser.add_argument("--num_frames", type=int,   default=49)
    parser.add_argument("--batch_size", type=int,   default=12)
    parser.add_argument("--gpu",        type=str,   default="1")
    return parser.parse_args()


# ── Attention modules ─────────────────────────────────────────────────────────
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
        B, C, H, W, D = volume.shape
        x     = volume.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        feats = self.feature_extractor(x).reshape(B, D, -1)
        A     = F.softmax(self.attention_net(feats), dim=1)
        agg   = torch.bmm(A.permute(0, 2, 1), feats).squeeze(1)
        return self.classifier(agg).squeeze(1)


def load_model(ckpt_path, device):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
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
def make_slice_transform(image_size):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])


def load_volume(volume_dir, num_frames, slice_transform):
    slice_paths = sorted([
        os.path.join(volume_dir, f)
        for f in os.listdir(volume_dir)
        if f.lower().endswith('.png')
    ])
    slices = []
    for sp in slice_paths:
        img = Image.open(sp).convert('L')
        t   = v2.functional.to_image(img)
        t   = slice_transform(t)
        slices.append(t)

    if not slices:
        return None

    volume = torch.stack(slices, dim=-1)
    D = volume.shape[-1]
    if D < num_frames:
        volume = F.pad(volume, (0, num_frames - D))
    else:
        volume = volume[..., :num_frames]

    return volume


def get_volume_dirs(volume_root):
    dirs = []
    for name in sorted(os.listdir(volume_root)):
        path = os.path.join(volume_root, name)
        if os.path.isdir(path):
            dirs.append((name, path))
    return dirs


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    slice_transform = make_slice_transform(args.image_size)

    model = load_model(args.ckpt_path, device)
    print(f"Model loaded from {args.ckpt_path}")

    volume_dirs = get_volume_dirs(args.volume_dir)
    print(f"Found {len(volume_dirs)} volume subfolders in {args.volume_dir}")

    rows = []
    for i in tqdm(range(0, len(volume_dirs), args.batch_size), desc="Inferring", unit="batch"):
        batch_info    = volume_dirs[i:i + args.batch_size]
        batch_volumes = []
        batch_names   = []
        batch_paths   = []

        for name, vdir in batch_info:
            vol = load_volume(vdir, args.num_frames, slice_transform)
            if vol is None:
                print(f"  Skipping {vdir} — no PNGs found")
                continue
            batch_volumes.append(vol)
            batch_names.append(name)
            batch_paths.append(vdir)

        if not batch_volumes:
            continue

        volumes = torch.stack(batch_volumes).to(device)
        with torch.no_grad():
            logits = model(volumes)
            probs  = torch.sigmoid(logits).cpu().numpy()

        preds = (probs >= args.threshold).astype(int)
        for name, vdir, prob, pred in zip(batch_names, batch_paths, probs, preds):
            rows.append({
                'volume_name': name,
                'filepath':    vdir,
                'probability': round(float(prob), 4),
                'prediction':  int(pred),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(df)} predictions to {args.output_csv}")
    print(f"  GA+: {df['prediction'].sum()}  |  GA-: {(df['prediction'] == 0).sum()}")


if __name__ == "__main__":
    main()