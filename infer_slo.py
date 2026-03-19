import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH  = "/scratch90/russ/GAclassifier2/SLO/20260317_231427_imbalanced/checkpoints/best_model.ckpt"
IMAGE_DIR  = "/scratch90/russ/GAclassifier2/SLO/images"
OUTPUT_CSV = "/scratch90/russ/GAclassifier2/SLO/slo_predictions.csv"
THRESHOLD  = 0.5
IMAGE_SIZE = 512
BATCH_SIZE = 16

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith('.png'):
            paths.append(os.path.join(image_dir, fname))
    return paths


def run_inference(model, paths, device):
    all_probs = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Inferring", unit="batch"):
        batch_paths = paths[i:i + BATCH_SIZE]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(CKPT_PATH, device)
    print(f"Model loaded from {CKPT_PATH}")

    paths = get_png_paths(IMAGE_DIR)
    print(f"Found {len(paths)} PNGs in {IMAGE_DIR}")

    probs = run_inference(model, paths, device)
    preds = (probs >= THRESHOLD).astype(int)

    df = pd.DataFrame({
        'filepath':    paths,
        'probability': np.round(probs, 4),
        'prediction':  preds,
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} predictions to {OUTPUT_CSV}")
    print(f"  GA+: {preds.sum()}  |  GA-: {(preds == 0).sum()}")


if __name__ == "__main__":
    main()