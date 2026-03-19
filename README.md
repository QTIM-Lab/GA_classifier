# GA Classifier Inference

Inference scripts for the three binary geographic atrophy (GA) classifiers: FAF, SLO, and OCT.

## Scripts

| File | Description |
|---|---|
| `infer_2d.py` | Inference script for 2D modalities (FAF and SLO) |
| `infer_oct.py` | Inference script for OCT volumes (MIL-based) |
| `infer_2D.sh` | Config + launcher for FAF and SLO inference |
| `infer_oct.sh` | Config + launcher for OCT inference |

## Usage
```bash
bash infer_2D.sh
bash infer_oct.sh
```

To change any parameter (checkpoint path, image directory, threshold, etc.), edit the
corresponding `.sh` file directly.

## Arguments

### `infer_2d.py` (FAF / SLO)

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` | required | Path to Lightning checkpoint (`.ckpt`) |
| `--image_dir` | required | Directory of input `.png` images |
| `--output_csv` | required | Path to write predictions CSV |
| `--threshold` | `0.5` | Classification threshold on sigmoid probability |
| `--image_size` | `512` | Image resize dimension (square) |
| `--batch_size` | `32` | Images per batch |
| `--gpu` | `0` | CUDA device index (`CUDA_VISIBLE_DEVICES`) |

### `infer_oct.py` (OCT)

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` | required | Path to Lightning checkpoint (`.ckpt`) |
| `--volume_dir` | required | Directory of volume subfolders (each containing `.png` slices) |
| `--output_csv` | required | Path to write predictions CSV |
| `--threshold` | `0.5` | Classification threshold on sigmoid probability |
| `--image_size` | `256` | Slice resize dimension (square) |
| `--num_frames` | `49` | Number of B-scan slices per volume (pads or truncates to this) |
| `--batch_size` | `12` | Volumes per batch |
| `--gpu` | `1` | CUDA device index (`CUDA_VISIBLE_DEVICES`) |

## Input Format

- **FAF / SLO**: flat directory of `.png` files
- **OCT**: directory of volume subfolders, each containing `.png` B-scan slices named
  in sorted order (e.g., `slice_000.png`, `slice_001.png`, ...)

## Output Format

All scripts write a CSV with the following columns:

| Column | Description |
|---|---|
| `filepath` | Absolute path to the input image or volume folder |
| `volume_name` | (OCT only) Name of the volume subfolder |
| `probability` | Sigmoid output of the model (0–1) |
| `prediction` | Binary label: `1` = GA+, `0` = GA− |

## Models

| Modality | Architecture |
|---|---|
| FAF | ResNet50 + dropout head |
| SLO | ResNet50 + dropout head |
| OCT | ResNet18 feature extractor + gated attention MIL |