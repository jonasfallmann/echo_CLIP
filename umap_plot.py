#!/usr/bin/env python3
"""
UMAP projection of EchoCLIP video features.

Extracts latent representations from the frozen EchoCLIP model,
pools them into per-video embeddings, computes UMAP, and plots
coloured by diagnosis/class.

Adapted from ref_Umap.py for the EchoCLIP repo.

Usage (config-based, same format as probe_train_mr.py):
    python umap_plot.py --config configs/probe_grid_mr.yaml

Usage (direct CLI):
    python umap_plot.py \
        --model hf-hub:mkaichristensen/echo-clip \
        --train-csv /path/to/train.csv \
        --val-csv /path/to/val.csv \
        --output-dir umap_output

UMAP overrides:
    --umap-n-neighbors 15
    --umap-min-dist 0.05
    --umap-metric euclidean
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import umap
import yaml
from open_clip import create_model_and_transforms
from tqdm import tqdm

from utils import CLASS_ORDER, normalize_label, read_avi, resolve_video_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirror probe_train_mr.py)
# ---------------------------------------------------------------------------

def expected_num_frames(max_frames: int, frame_step: int) -> int:
    import math
    return max(1, int(math.ceil(max_frames / max(1, frame_step))))


def sample_frames_fixed(video_np: np.ndarray, max_frames: int = 40, frame_step: int = 2) -> np.ndarray:
    """Sample frames from a video numpy array [T, H, W, C] with fixed stride + padding."""
    tgt_len = expected_num_frames(max_frames=max_frames, frame_step=frame_step)
    frames = video_np[0 : min(max_frames, len(video_np)) : frame_step]
    if len(frames) == 0:
        frames = video_np[:1]

    if len(frames) < tgt_len:
        pad_count = tgt_len - len(frames)
        pad = np.repeat(frames[-1][None, ...], pad_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:tgt_len]
    return frames


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(
    csv_path: Path,
    video_path_col: str,
    label_col: str,
    subject_id_col: str,
    borderline_policy: str,
) -> tuple[list[dict], int]:
    """Load video records from a CSV file.

    Returns (records, dropped_count).  Each record has:
        video_path, subject_id, label_id (0..3), label_name
    """
    records = []
    dropped = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = normalize_label(row[label_col], borderline_policy=borderline_policy)
            if label_name is None:
                dropped += 1
                continue

            video_path = resolve_video_path(row[video_path_col], csv_path)
            if not video_path.exists():
                dropped += 1
                continue

            subject_id = str(row.get(subject_id_col, "")).strip() or str(video_path)
            records.append(
                {
                    "video_path": str(video_path),
                    "subject_id": subject_id,
                    "label_id": CLASS_ORDER.index(label_name),
                    "label_name": label_name,
                }
            )

    return records, dropped


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    records: list[dict],
    preprocess,
    device: torch.device,
    max_frames: int,
    frame_step: int,
    batch_size: int,
) -> dict:
    """
    Iterate over records in batches, extract per-frame embeddings with
    model.encode_image, then mean-pool across frames → one D-dim vector per video.

    Returns:
        {
            "embeddings":  np.ndarray  [N, D],
            "labels":      np.ndarray  [N],
            "patient_ids": list[str],
        }
    """
    all_embeddings = []
    all_labels = []
    all_patient_ids = []

    model_dtype = next(model.parameters()).dtype
    num_batches = (len(records) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting features", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batch_records = records[start:end]

        # -- Load & preprocess frames for all videos in this batch --
        batch_frame_list: list[torch.Tensor] = []
        batch_n_frames: list[int] = []
        for rec in batch_records:
            video_np = read_avi(rec["video_path"], (224, 224))
            if len(video_np) == 0:
                tgt_len = expected_num_frames(max_frames, frame_step)
                video_np = np.zeros((tgt_len, 224, 224, 3), dtype=np.uint8)
            frames_np = sample_frames_fixed(video_np, max_frames=max_frames, frame_step=frame_step)
            pixel_values = torch.stack([preprocess(T.ToPILImage()(f)) for f in frames_np], dim=0)
            batch_frame_list.append(pixel_values)
            batch_n_frames.append(pixel_values.shape[0])

        # -- Forward all frames at once (faster) --
        all_pixels = torch.cat(batch_frame_list, dim=0).to(device, dtype=model_dtype, non_blocking=True)
        frame_embeddings = model.encode_image(all_pixels)   # [total_frames, D]
        frame_embeddings = F.normalize(frame_embeddings.float(), dim=-1)

        # -- Split back per video and mean-pool --
        idx = 0
        for i, rec in enumerate(batch_records):
            n = batch_n_frames[i]
            video_frames = frame_embeddings[idx : idx + n]
            idx += n

            video_emb = video_frames.mean(dim=0).cpu().numpy()

            all_embeddings.append(video_emb)
            all_labels.append(rec["label_id"])
            all_patient_ids.append(rec["subject_id"])

    logger.info(
        "Extracted %d video embeddings (dim=%d)",
        len(all_embeddings),
        all_embeddings[0].shape[0],
    )
    return {
        "embeddings": np.stack(all_embeddings, axis=0),
        "labels": np.array(all_labels),
        "patient_ids": all_patient_ids,
    }


# ---------------------------------------------------------------------------
# UMAP + Plot (from ref_Umap.py, adapted)
# ---------------------------------------------------------------------------

def compute_and_plot(
    all_features: dict[str, dict],
    class_names: list[str],
    output_dir: str,
    umap_kwargs: dict | None = None,
):
    """
    all_features:  {set_name: {"embeddings": [N,D], "labels": [N]}, ...}
    class_names:   label-index → human name
    """
    if umap_kwargs is None:
        umap_kwargs = dict(n_neighbors=30, min_dist=0.1, n_components=2,
                           metric="cosine", random_state=42)

    set_labels = sorted(all_features.keys())

    # -- collect
    all_emb = []
    all_lbl = []
    all_set = []
    for sname in set_labels:
        feats = all_features[sname]
        all_emb.append(feats["embeddings"])
        all_lbl.append(feats["labels"])
        all_set.append(np.full(len(feats["labels"]), sname, dtype=object))

    X = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_lbl, axis=0)
    sets = np.concatenate(all_set, axis=0)

    logger.info(f"Total samples for UMAP: {len(X)} (dim={X.shape[1]})")
    logger.info(f"UMAP kwargs: {umap_kwargs}")

    # -- run UMAP
    reducer = umap.UMAP(**umap_kwargs)
    embedding_2d = reducer.fit_transform(X)

    unique = np.unique(y)
    n_classes = len(unique)

    # Custom class colours (fall back to tab20 for extra classes)
    custom_colours = ["#27AE60", "#F1C40F", "#E67E22", "#E74C3C"]
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colours = [
        custom_colours[i] if i < len(custom_colours) else cmap(i % 20)
        for i in range(n_classes)
    ]

    # -------- Figure 1: two panels (class + split) --------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    set_markers = {"train": "o", "val": "s", "test": "^"}

    # Panel 1: colour by class, marker by set
    ax = axes[0]
    for lbl in unique:
        for sname in set_labels:
            mask = (y == lbl) & (sets == sname)
            if not mask.any():
                continue
            name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[colours[lbl]], marker=set_markers.get(sname, "o"),
                s=25, alpha=0.7, edgecolors="none",
                label=f"{name} ({sname})",
            )
    ax.set_title(
        f"UMAP by Class & Split  "
        f"(n_neighbors={umap_kwargs['n_neighbors']}, min_dist={umap_kwargs['min_dist']})"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, markerscale=1.5)

    # Panel 2: colour by set
    ax = axes[1]
    set_colours = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for sname in set_labels:
        mask = sets == sname
        if not mask.any():
            continue
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=set_colours.get(sname, "gray"), marker="o",
            s=25, alpha=0.7, edgecolors="none", label=sname,
        )
    ax.set_title("UMAP by Split")
    ax.legend(fontsize=12)

    plt.tight_layout()

    path1 = os.path.join(output_dir, "umap_projection.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path1}")
    plt.close(fig)

    # -------- Figure 2: class-only --------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    for lbl in unique:
        mask = y == lbl
        if not mask.any():
            continue
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax2.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[colours[lbl]], s=50, alpha=0.9, edgecolors="none", label=name,
        )
    ax2.legend(fontsize=14)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "umap_projection_classes.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path2}")
    plt.close(fig2)

    # -------- Save raw data --------
    np.savez(
        os.path.join(output_dir, "umap_data.npz"),
        embedding_2d=embedding_2d,
        labels=y,
        sets=sets,
        class_names=np.array(class_names[:n_classes]),
    )
    logger.info(f"Saved umap_data.npz to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UMAP projection of EchoCLIP video features")

    # -- Config file (same format as probe_train_mr.py configs)
    p.add_argument("--config", type=Path, default=None,
                   help="Path to YAML config (probe training format)")

    # -- Model
    p.add_argument("--model", type=str, default=None,
                   help="EchoCLIP model name (default: hf-hub:mkaichristensen/echo-clip)")

    # -- Data
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv",   type=Path, default=None)
    p.add_argument("--test-csv",  type=Path, default=None)
    p.add_argument("--video-path-col",  type=str, default=None)
    p.add_argument("--label-col",       type=str, default=None)
    p.add_argument("--subject-id-col",  type=str, default=None)
    p.add_argument("--borderline-policy", type=str, choices=["round_up", "round_down", "drop"],
                   default=None)
    p.add_argument("--max-frames",  type=int, default=None)
    p.add_argument("--frame-step",  type=int, default=None)
    p.add_argument("--batch-size",  type=int, default=None)
    p.add_argument("--output-dir",  type=str, default=None)

    # -- UMAP overrides
    p.add_argument("--umap-n-neighbors", type=int,   default=None)
    p.add_argument("--umap-min-dist",    type=float, default=None)
    p.add_argument("--umap-metric",      type=str,   default=None)

    # -- Class names (space-separated, e.g.: "None / Trace" Mild Moderate Severe)
    #     Label-id order (0..3).  Also settable via config: umap.class_names
    p.add_argument("--class-names", nargs=4, default=None,
                   help="Four class names in label-id order (0=normal, 1=mild, 2=moderate, 3=severe)")

    # -- Device
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def main():
    args = parse_args()

    # -- Defaults --
    defaults = {
        "model_name": "hf-hub:mkaichristensen/echo-clip",
        "video_path_col": "video_filename",
        "label_col": "label",
        "subject_id_col": "patient_id",
        "borderline_policy": "round_up",
        "max_frames": 32,
        "frame_step": 1,
        "batch_size": 10,
        "output_dir": "umap_output",
    }

    set_paths: dict[str, Path] = {}

    # 1) Load config if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with args.config.open("r") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        logging_cfg = cfg.get("logging", {})

        defaults["model_name"]       = model_cfg.get("name", defaults["model_name"])
        defaults["video_path_col"]   = data_cfg.get("video_path_col", defaults["video_path_col"])
        defaults["label_col"]        = data_cfg.get("label_col", defaults["label_col"])
        defaults["subject_id_col"]   = data_cfg.get("subject_id_col", defaults["subject_id_col"])
        defaults["borderline_policy"] = data_cfg.get("borderline_policy", defaults["borderline_policy"])
        defaults["max_frames"]       = int(data_cfg.get("max_frames", defaults["max_frames"]))
        defaults["frame_step"]       = int(data_cfg.get("frame_step", defaults["frame_step"]))
        defaults["batch_size"]       = int(data_cfg.get("batch_size", defaults["batch_size"]))
        defaults["output_dir"]       = os.path.join(
            str(logging_cfg.get("output_dir", defaults["output_dir"])), "umap"
        )

        # Optional umap section in config
        umap_cfg = cfg.get("umap", {})
        if umap_cfg.get("class_names"):
            defaults["class_names"] = list(umap_cfg["class_names"])

        if data_cfg.get("train_csv"):
            set_paths["train"] = Path(data_cfg["train_csv"])
        if data_cfg.get("val_csv"):
            set_paths["val"] = Path(data_cfg["val_csv"])
        if data_cfg.get("test_csv"):
            set_paths["test"] = Path(data_cfg["test_csv"])

    # 2) CLI overrides
    if args.model:
        defaults["model_name"] = args.model
    if args.output_dir:
        defaults["output_dir"] = args.output_dir
    for key in ("video_path_col", "label_col", "subject_id_col", "borderline_policy"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            defaults[key] = val
    for key in ("max_frames", "frame_step", "batch_size"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            defaults[key] = val

    if args.train_csv:
        set_paths["train"] = args.train_csv
    if args.val_csv:
        set_paths["val"] = args.val_csv
    if args.test_csv:
        set_paths["test"] = args.test_csv

    if not set_paths:
        logger.error("No datasets specified. Use --config or --train-csv / --val-csv / --test-csv.")
        sys.exit(1)

    # -- Resolve final values --
    model_name       = defaults["model_name"]
    video_path_col   = defaults["video_path_col"]
    label_col        = defaults["label_col"]
    subject_id_col   = defaults["subject_id_col"]
    borderline_policy = defaults["borderline_policy"]
    max_frames       = defaults["max_frames"]
    frame_step       = defaults["frame_step"]
    batch_size       = defaults["batch_size"]
    output_dir       = defaults["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # -- Class names: CLI > config umap section > CLASS_ORDER
    class_names = defaults.get("class_names", list(CLASS_ORDER))
    if args.class_names:
        class_names = list(args.class_names)

    logger.info(f"Class names: {class_names}")

    # UMAP kwargs
    umap_kwargs = {
        "n_neighbors":  args.umap_n_neighbors or 30,
        "min_dist":     args.umap_min_dist    or 0.1,
        "metric":       args.umap_metric      or "cosine",
        "n_components": 2,
        "random_state": 42,
    }

    # -- Device --
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_name}")
    logger.info(f"UMAP sets: {list(set_paths.keys())}")
    logger.info(f"UMAP kwargs: {umap_kwargs}")
    logger.info(f"Output dir: {output_dir}")

    # -- Load model --
    logger.info("Loading EchoCLIP model...")
    precision = "bf16" if device.type == "cuda" else "fp32"
    model, _, preprocess_val = create_model_and_transforms(
        model_name, precision=precision, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("Model loaded.")

    # -- Extract features for each set --
    all_features = {}
    for sname, csv_path in set_paths.items():
        logger.info(f"Loading {sname} from {csv_path}")
        records, dropped = load_records(
            csv_path, video_path_col, label_col, subject_id_col, borderline_policy,
        )
        logger.info(f"  {sname}: {len(records)} videos (dropped: {dropped})")

        if not records:
            logger.warning(f"  No records for {sname}, skipping.")
            continue

        feats = extract_features(
            model=model,
            records=records,
            preprocess=preprocess_val,
            device=device,
            max_frames=max_frames,
            frame_step=frame_step,
            batch_size=batch_size,
        )
        all_features[sname] = feats
        logger.info(
            "  %s: %d samples, %d unique classes",
            sname,
            feats["embeddings"].shape[0],
            len(np.unique(feats["labels"])),
        )

    if not all_features:
        logger.error("No features extracted. Exiting.")
        sys.exit(1)

    # -- UMAP + plot --
    compute_and_plot(all_features, class_names, output_dir, umap_kwargs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
