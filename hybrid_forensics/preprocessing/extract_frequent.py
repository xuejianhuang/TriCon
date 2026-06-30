import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import pywt
except ImportError:
    pywt = None

try:
    from scipy.fft import dct, idct as idct_
except ImportError:
    dct = None
    idct_ = None


def wavelet_decompose(
    features: np.ndarray,
    wavelet: str = "db4",
    mode: str = "symmetric",
):
    T, D = features.shape
    cA_cols = []
    cD_cols = []

    for d in range(D):
        col = features[:, d]
        coeffs = pywt.wavedec(col, wavelet=wavelet, level=1, mode=mode)
        cA_cols.append(coeffs[0])
        cD_cols.append(coeffs[1])

    cA = np.stack(cA_cols, axis=1).astype(np.float32)
    cD = np.stack(cD_cols, axis=1).astype(np.float32)
    return cA, cD


def wavelet_reconstruct(
    cA: np.ndarray,
    cD: np.ndarray,
    wavelet: str = "db4",
    mode: str = "symmetric",
):
    D = cA.shape[1]
    rec_cols = []

    for d in range(D):
        coeffs = [cA[:, d], cD[:, d]]
        rec = pywt.waverec(coeffs, wavelet=wavelet, mode=mode)
        rec_cols.append(rec)

    reconstructed = np.stack(rec_cols, axis=1).astype(np.float32)
    return reconstructed


def dct_decompose(
    features: np.ndarray,
    cutoff_ratio: float = 0.5,
):
    if dct is None:
        raise ImportError(
            "scipy required for DCT mode. Install with: pip install scipy"
        )
    T, D = features.shape
    dct_coeffs = dct(features, axis=0, norm="ortho")
    cutoff = max(1, int(T * cutoff_ratio))
    low_freq = dct_coeffs[:cutoff, :]
    high_freq = dct_coeffs[cutoff:, :]
    return low_freq, high_freq, dct_coeffs, cutoff


def dct_reconstruct(
    low_freq: np.ndarray,
    high_freq: np.ndarray,
    original_len: int,
):
    if idct_ is None:
        raise ImportError(
            "scipy required for DCT mode. Install with: pip install scipy"
        )
    full_coeffs = np.concatenate([low_freq, high_freq], axis=0)
    return idct_(full_coeffs, axis=0, n=original_len, norm="ortho")


def _block_avg_pool2d(arr: np.ndarray, grid: int):
    H, W = arr.shape
    bh = max(1, H // grid)
    bw = max(1, W // grid)
    h_trim = bh * grid
    w_trim = bw * grid
    trimmed = arr[:h_trim, :w_trim]
    return trimmed.reshape(grid, bh, grid, bw).mean(axis=(1, 3)).astype(np.float32)


def pixel_dwt_extract_features(
    video_path: str,
    wavelet: str = "db4",
    mlp: nn.Module = None,
    max_frames: int = 300,
    resize: int = 256,
):
    if pywt is None:
        raise ImportError("PyWavelets required.  Install with: pip install PyWavelets")

    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for pixel_dwt mode")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames and total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = None

    frames_raw = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if indices is not None:
            if frame_idx not in set(indices):
                frame_idx += 1
                continue
        if resize:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames_raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))
        frame_idx += 1
    cap.release()

    if not frames_raw:
        raise RuntimeError(f"No frames read from {video_path}")

    frames = np.stack(frames_raw, axis=0)
    del frames_raw
    T, H, W, C = frames.shape
    grid_size = 8
    vals_per_subband = grid_size * grid_size
    total_subbands = 12
    feat_dim = vals_per_subband * total_subbands

    frame_feats = np.zeros((T, feat_dim), dtype=np.float32)
    for t in range(T):
        parts = []
        frame_t = frames[t]
        for c in range(C):
            coeffs = pywt.dwt2(frame_t[:, :, c], wavelet=wavelet, mode="symmetric")
            cA, (cH, cV, cD) = coeffs
            for subband in (cA, cH, cV, cD):
                parts.append(_block_avg_pool2d(subband, grid_size).ravel())
        frame_feats[t] = np.concatenate(parts)

    del frames

    cA, cD = wavelet_decompose(frame_feats, wavelet=wavelet)

    if mlp is None:
        mlp = LowFreqSuppressMLP(dim=feat_dim)
    mlp.eval()
    with torch.no_grad():
        cA_tensor = torch.from_numpy(cA)
        cA_transformed = mlp(cA_tensor).numpy().astype(np.float32)

    frequent = wavelet_reconstruct(cA_transformed, cD, wavelet=wavelet)

    if frequent.shape[0] > T:
        frequent = frequent[:T, :]
    elif frequent.shape[0] < T:
        frequent = np.pad(frequent, ((0, T - frequent.shape[0]), (0, 0)), mode="edge")

    return {
        "frequent": torch.from_numpy(frequent),
        "avg_logit": 0.0,
        "label": -1,
        "mlp_state": mlp.state_dict(),
        "transform": "pixel_dwt",
    }


class LowFreqSuppressMLP(nn.Module):

    def __init__(self, dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_frequent_from_dynamic(
    dynamic_path: str,
    wavelet: str = "db4",
    lowfreq_mlp: Optional[nn.Module] = None,
):
    data = torch.load(dynamic_path, map_location="cpu", weights_only=False)
    temporal = data["temporal"]

    if isinstance(temporal, torch.Tensor):
        temporal_np = temporal.numpy().astype(np.float32)
    else:
        temporal_np = np.asarray(temporal, dtype=np.float32)

    original_len = temporal_np.shape[0]

    cA, cD = wavelet_decompose(temporal_np, wavelet=wavelet)

    if lowfreq_mlp is None:
        lowfreq_mlp = LowFreqSuppressMLP(dim=cA.shape[1])

    with torch.no_grad():
        cA_tensor = torch.from_numpy(cA)
        cA_transformed_tensor = lowfreq_mlp(cA_tensor)
        cA_transformed = cA_transformed_tensor.numpy().astype(np.float32)

    frequent = wavelet_reconstruct(cA_transformed, cD, wavelet=wavelet)

    if frequent.shape[0] > original_len:
        frequent = frequent[:original_len, :]
    elif frequent.shape[0] < original_len:
        pad_len = original_len - frequent.shape[0]
        frequent = np.pad(frequent, ((0, pad_len), (0, 0)), mode="edge")

    return {
        "frequent": torch.from_numpy(frequent),
        "cA": torch.from_numpy(cA_transformed),
        "cD": torch.from_numpy(cD),
        "avg_logit": data.get("avg_logit", 0.0),
        "label": data.get("label", -1),
        "mlp_state": lowfreq_mlp.state_dict(),
    }


def extract_frequent_from_dynamic_dct(
    dynamic_path: str,
    cutoff_ratio: float = 0.5,
    lowfreq_mlp: Optional[nn.Module] = None,
):
    data = torch.load(dynamic_path, map_location="cpu", weights_only=False)
    temporal = data["temporal"]

    if isinstance(temporal, torch.Tensor):
        temporal_np = temporal.numpy().astype(np.float32)
    else:
        temporal_np = np.asarray(temporal, dtype=np.float32)

    original_len = temporal_np.shape[0]

    low_freq, high_freq, _dct_full, cutoff = dct_decompose(
        temporal_np, cutoff_ratio=cutoff_ratio,
    )

    if lowfreq_mlp is None:
        lowfreq_mlp = LowFreqSuppressMLP(dim=low_freq.shape[1])

    with torch.no_grad():
        cA_tensor = torch.from_numpy(low_freq)
        cA_transformed_tensor = lowfreq_mlp(cA_tensor)
        cA_transformed = cA_transformed_tensor.numpy().astype(np.float32)

    frequent = dct_reconstruct(cA_transformed, high_freq, original_len)

    if frequent.shape[0] > original_len:
        frequent = frequent[:original_len, :]
    elif frequent.shape[0] < original_len:
        pad_len = original_len - frequent.shape[0]
        frequent = np.pad(frequent, ((0, pad_len), (0, 0)), mode="edge")

    return {
        "frequent": torch.from_numpy(frequent),
        "low_freq_coeffs": torch.from_numpy(cA_transformed),
        "high_freq_coeffs": torch.from_numpy(high_freq),
        "avg_logit": data.get("avg_logit", 0.0),
        "label": data.get("label", -1),
        "mlp_state": lowfreq_mlp.state_dict(),
    }


def extract_frequent_features(
    video_list_file: Optional[str] = None,
    feature_root: Optional[str] = None,
    video_root: Optional[str] = None,
    output_dir: str = "data/frequent_features",
    transform: str = "dwt",
    wavelet: str = "db4",
    dct_cutoff_ratio: float = 0.5,
    mlp_hidden_dim: int = 256,
    skip_existing: bool = True,
    max_frames: int = 300,
    resize: int = 256,
):
    if video_list_file is not None:
        with open(video_list_file, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        entries = []
        for line in raw_lines:
            parts = line.rsplit(None, 1) if ' ' in line else line.rsplit(',', 1)
            if len(parts) < 2:
                continue
            rel_path, label_str = parts
            video_base = os.path.splitext(rel_path)[0]
            entries.append((video_base, int(label_str), rel_path))
    elif feature_root is not None:
        entries = []
        if not os.path.isdir(feature_root):
            raise FileNotFoundError(f"Feature root not found: {feature_root}")
        for dirpath, _dirnames, filenames in os.walk(feature_root):
            if "dynamic_features.pt" in filenames:
                rel_dir = os.path.relpath(dirpath, feature_root)
                entries.append((rel_dir, -1, rel_dir))
        entries.sort()
    else:
        raise ValueError("One of --video_list or --feature_root must be provided.")

    if not entries:
        print("No samples found — nothing to process.")
        return

    print(f"Found {len(entries)} samples to process")
    print(f"  Transform: {transform}")
    if transform == "dwt":
        print(f"  Wavelet: {wavelet}")
    elif transform == "dct":
        print(f"  DCT cutoff ratio: {dct_cutoff_ratio}")
    elif transform == "pixel_dwt":
        print(f"  Wavelet: {wavelet}")
        print(f"  Video root: {video_root}")
    print(f"  Low-freq MLP hidden dim: {mlp_hidden_dim}")
    if transform == "pixel_dwt":
        print(f"  Max frames: {max_frames}")
        print(f"  Resize: {resize}")
    print(f"  Output:   {output_dir}")

    lowfreq_mlp = LowFreqSuppressMLP(dim=768, hidden_dim=mlp_hidden_dim)
    lowfreq_mlp.eval()

    success = 0
    skip = 0
    error = 0

    for video_base, label, rel_path in tqdm(entries, desc="Extracting frequent features"):
        if feature_root is not None:
            dyn_path = os.path.join(feature_root, video_base, "dynamic_features.pt")
        else:
            dyn_path = None

        out_dir = os.path.join(output_dir, video_base)
        out_path = os.path.join(out_dir, "frequent_features.pt")

        if skip_existing and os.path.exists(out_path):
            skip += 1
            continue

        if transform in ("dwt", "dct"):
            if not os.path.exists(dyn_path):
                error += 1
                print(f"  [skip] missing dynamic_features.pt for {video_base}")
                continue

        if transform == "pixel_dwt":
            if video_root is None:
                raise ValueError("--video_root required for pixel_dwt transform")
            video_path = os.path.join(video_root, rel_path)
            if not os.path.exists(video_path):
                error += 1
                print(f"  [skip] missing video for {video_base}")
                continue

        try:
            os.makedirs(out_dir, exist_ok=True)

            if transform == "dwt":
                freq_feat = extract_frequent_from_dynamic(
                    dyn_path, wavelet=wavelet, lowfreq_mlp=lowfreq_mlp,
                )
            elif transform == "dct":
                freq_feat = extract_frequent_from_dynamic_dct(
                    dyn_path, cutoff_ratio=dct_cutoff_ratio,
                    lowfreq_mlp=lowfreq_mlp,
                )
            elif transform == "pixel_dwt":
                video_path = os.path.join(video_root, rel_path)
                freq_feat = pixel_dwt_extract_features(
                    video_path, wavelet=wavelet, mlp=lowfreq_mlp,
                    max_frames=max_frames, resize=resize,
                )
            else:
                raise ValueError(f"Unknown transform: {transform}")

            if label >= 0:
                freq_feat["label"] = label
            freq_feat["transform"] = transform
            torch.save(freq_feat, out_path)
            success += 1
        except Exception as exc:
            error += 1
            print(f"  [error] {video_base}: {exc}")

    print(f"\nDone.  Success: {success}  |  Skipped: {skip}  |  Errors: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frequency features via wavelet transform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_list",
        default=None,
        help='Path to file_list.txt (format: "rel/path label"). '
        "If omitted, --feature_root is scanned for dynamic_features.pt files.",
    )
    parser.add_argument(
        "--feature_root",
        default=None,
        help="Root directory containing per-sample folders with dynamic_features.pt "
        "(e.g. data/cached_features/train).",
    )
    parser.add_argument(
        "--video_root",
        default=None,
        help="Root directory containing video files (required for --transform pixel_dwt).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/frequent_features/train",
        help="Output directory for frequent_features.pt files.",
    )
    parser.add_argument(
        "--transform",
        default="dwt",
        choices=["dwt", "dct", "pixel_dwt"],
        help="Frequency transform mode: "
        "dwt = feature-level wavelet (default), "
        "dct = DCT instead of wavelet, "
        "pixel_dwt = 2D wavelet on RGB pixel level.",
    )
    parser.add_argument(
        "--wavelet",
        default="db4",
        choices=["db1", "db2", "db4", "db8", "sym4", "sym5", "coif1", "bior2.2"],
        help="Wavelet family used for DWT (ignored for DCT mode).",
    )
    parser.add_argument(
        "--dct_cutoff_ratio",
        type=float,
        default=0.5,
        help="Fraction of DCT coefficients treated as low-frequency (DCT mode only).",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the LowFreqSuppressMLP used to transform "
        "low-frequency coefficients.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose frequent_features.pt already exists.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Overwrite existing frequent_features.pt files.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Max frames per video for pixel_dwt transform (default: 300, "
        "uniformly sampled). 300 frames @ 25fps = 12 seconds coverage. "
        "Reduce this if you encounter OOM.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize frames to (resize, resize) before processing for pixel_dwt "
        "(default: 256). The final feature uses 8x8 block averaging, so "
        "resolutions above 128 add minimal value but consume much more memory.",
    )

    args = parser.parse_args()

    if args.video_list is None and args.feature_root is None:
        parser.error("One of --video_list or --feature_root must be specified.")
    if args.transform == "pixel_dwt" and args.video_root is None:
        parser.error("--video_root is required for --transform pixel_dwt")

    extract_frequent_features(
        video_list_file=args.video_list,
        feature_root=args.feature_root,
        video_root=args.video_root,
        output_dir=args.output_dir,
        transform=args.transform,
        wavelet=args.wavelet,
        dct_cutoff_ratio=args.dct_cutoff_ratio,
        mlp_hidden_dim=args.mlp_hidden_dim,
        skip_existing=args.skip_existing,
        max_frames=args.max_frames,
        resize=args.resize,
    )


if __name__ == "__main__":
    main()
