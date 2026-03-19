import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="2D GA classifier inference (FAF or SLO)")
    parser.add_argument("--ckpt_path",  type=str,   required=True)
    parser.add_argument("--image_dir",  type=str,   required=True)
    parser.add_argument("--output_csv", type=str,   required=True)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--image_size", type=int,   default=512)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--gpu",        type=str,   default="0")
    return parser.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────
class GAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 1))
        self.model = backbone

    def forward(self, x):
        return self.model(x).squeeze(1)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    model = GAClassifier()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_png_paths(image_dir):
    return [
        os.path.join(image_dir, fname)
        for fname in sorted(os.listdir(image_dir))
        if fname.lower().endswith('.png')
    ]


def run_inference(model, paths, transform, batch_size, device):
    all_probs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Inferring", unit="batch"):
        batch_paths = paths[i:i + batch_size]
        tensors = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.ckpt_path, device)
    print(f"Model loaded from {args.ckpt_path}")

    paths = get_png_paths(args.image_dir)
    print(f"Found {len(paths)} PNGs in {args.image_dir}")

    probs = run_inference(model, paths, transform, args.batch_size, device)
    preds = (probs >= args.threshold).astype(int)

    df = pd.DataFrame({
        'filepath':    paths,
        'probability': np.round(probs, 4),
        'prediction':  preds,
    })
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(df)} predictions to {args.output_csv}")
    print(f"  GA+: {preds.sum()}  |  GA-: {(preds == 0).sum()}")


if __name__ == "__main__":
    main()