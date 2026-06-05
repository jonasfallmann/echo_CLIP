import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from open_clip import create_model_and_transforms
from torch.utils.data import DataLoader, Dataset

from attentive_pooler import AttentiveClassifier
from utils import CLASS_ORDER, normalize_label, read_avi, resolve_video_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a frozen Echo-CLIP encoder and a trained probe.")

    # Config argument
    parser.add_argument("--config", type=Path, default=None, help="Path to the training YAML config file.")

    # Data & Checkpoint (Required)
    parser.add_argument("--test-csv", type=Path, required=True, help="Path to the test CSV file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained probe .pt checkpoint.")

    # Optional CLI overrides
    parser.add_argument("--head-idx", type=int, default=None, help="Specific probe head index. Defaults to best head.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--output-dir", type=Path, default=Path("probe_inference_outputs"))

    return parser.parse_args()


def load_inference_config(args):
    """Merges YAML config (if provided) with CLI arguments."""
    # Default fallbacks
    cfg = {
        "model_name": "hf-hub:mkaichristensen/echo-clip",
        "device": "cuda",
        "video_path_col": "video_path",
        "label_col": "label",
        "subject_id_col": "subject_id",
        "borderline_policy": "round_up",
        "max_frames": 40,
        "frame_step": 2,
        "batch_size": 8,
        "num_workers": 4,
        "probe_depth": 1,
        "probe_num_heads": 8,
        "probe_mlp_ratio": 4.0,
    }

    # 1. Pull architecture and data parameters from training YAML
    if args.config:
        logger.info(f"Loading parameters from config: {args.config}")
        with args.config.open("r") as f:
            yaml_cfg = yaml.safe_load(f)

        cfg["model_name"] = yaml_cfg.get("model", {}).get("name", cfg["model_name"])

        data_cfg = yaml_cfg.get("data", {})
        cfg["video_path_col"] = data_cfg.get("video_path_col", cfg["video_path_col"])
        cfg["label_col"] = data_cfg.get("label_col", cfg["label_col"])
        cfg["subject_id_col"] = data_cfg.get("subject_id_col", cfg["subject_id_col"])
        cfg["borderline_policy"] = data_cfg.get("borderline_policy", cfg["borderline_policy"])
        cfg["max_frames"] = data_cfg.get("max_frames", cfg["max_frames"])
        cfg["frame_step"] = data_cfg.get("frame_step", cfg["frame_step"])
        cfg["batch_size"] = data_cfg.get("batch_size", cfg["batch_size"])
        cfg["num_workers"] = data_cfg.get("num_workers", cfg["num_workers"])

        probe_cfg = yaml_cfg.get("probe", {})
        cfg["probe_depth"] = probe_cfg.get("depth", cfg["probe_depth"])
        cfg["probe_num_heads"] = probe_cfg.get("num_heads", cfg["probe_num_heads"])
        cfg["probe_mlp_ratio"] = probe_cfg.get("mlp_ratio", cfg["probe_mlp_ratio"])

    # 2. CLI arguments take ultimate precedence
    cfg["test_csv"] = args.test_csv
    cfg["checkpoint"] = args.checkpoint
    cfg["head_idx"] = args.head_idx
    cfg["output_dir"] = args.output_dir

    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    return cfg


def expected_num_frames(max_frames, frame_step):
    return max(1, int(math.ceil(max_frames / max(1, frame_step))))


def sample_frames_fixed(video_np, max_frames=40, frame_step=2):
    tgt_len = expected_num_frames(max_frames=max_frames, frame_step=frame_step)
    frames = video_np[0: min(max_frames, len(video_np)): frame_step]
    if len(frames) == 0:
        frames = video_np[:1]

    if len(frames) < tgt_len:
        pad_count = tgt_len - len(frames)
        pad = np.repeat(frames[-1][None, ...], pad_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:tgt_len]

    return frames


def load_records(csv_path, video_path_col, label_col, subject_id_col, borderline_policy):
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
                }
            )

    return records, dropped


class EchoClipProbeDataset(Dataset):
    def __init__(self, records, preprocess, max_frames, frame_step):
        self.records = records
        self.preprocess = preprocess
        self.max_frames = max_frames
        self.frame_step = frame_step

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = read_avi(rec["video_path"], (224, 224))
        if len(frames) == 0:
            raise RuntimeError(f"No frames found in video: {rec['video_path']}")

        frames = sample_frames_fixed(frames, max_frames=self.max_frames, frame_step=self.frame_step)
        pixel_values = torch.stack([self.preprocess(T.ToPILImage()(f)) for f in frames], dim=0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(rec["label_id"], dtype=torch.long),
            "subject_id": rec["subject_id"],
            "video_path": rec["video_path"],
        }


def extract_video_tokens(model, pixel_values, device):
    bsz, nframes = pixel_values.shape[:2]
    flat_pixels = pixel_values.reshape(-1, *pixel_values.shape[2:])

    model_dtype = next(model.parameters()).dtype
    flat_pixels = flat_pixels.to(device=device, dtype=model_dtype, non_blocking=True)

    with torch.no_grad():
        frame_embeddings = model.encode_image(flat_pixels)
        frame_embeddings = F.normalize(frame_embeddings.float(), dim=-1)

    return frame_embeddings.reshape(bsz, nframes, -1)


def main():
    args = parse_args()
    cfg = load_inference_config(args)

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load the frozen Echo-CLIP encoder
    logger.info(f"Loading encoder: {cfg['model_name']}")
    precision = "bf16" if device.type == "cuda" else "fp32"
    model, _, preprocess_val = create_model_and_transforms(cfg["model_name"], precision=precision, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get embed_dim using a dummy pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device).to(
            dtype=torch.bfloat16 if precision == "bf16" else torch.float32)
        embed_dim = int(model.encode_image(dummy).shape[-1])

    # 2. Load Dataset and DataLoader
    logger.info(f"Loading test data from {cfg['test_csv']}")
    records, dropped = load_records(
        cfg["test_csv"],
        cfg["video_path_col"],
        cfg["label_col"],
        cfg["subject_id_col"],
        cfg["borderline_policy"]
    )
    if not records:
        raise RuntimeError("No usable rows in CSV after filtering.")

    logger.info(f"Found {len(records)} test videos (dropped: {dropped})")

    dataset = EchoClipProbeDataset(
        records=records,
        preprocess=preprocess_val,
        max_frames=cfg["max_frames"],
        frame_step=cfg["frame_step"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Load Checkpoint and Probe
    logger.info(f"Loading probe checkpoint from {cfg['checkpoint']}")
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu")

    saved_state_dicts = checkpoint.get("probe_state_dicts", [])
    if not saved_state_dicts:
        raise KeyError("Checkpoint does not contain `probe_state_dicts`.")

    head_idx = cfg["head_idx"] if cfg["head_idx"] is not None else checkpoint.get("best_head_idx", 0)
    if head_idx < 0 or head_idx >= len(saved_state_dicts):
        logger.warning(f"Head index {head_idx} is out of bounds for {len(saved_state_dicts)} heads. Falling back to 0.")
        head_idx = 0

    logger.info(f"Using probe head {head_idx} (from {len(saved_state_dicts)} available heads).")

    probe = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=cfg["probe_num_heads"],
        depth=cfg["probe_depth"],
        mlp_ratio=cfg["probe_mlp_ratio"],
        num_classes=len(CLASS_ORDER),
        use_activation_checkpointing=False,
    ).to(device)

    probe.load_state_dict(saved_state_dicts[head_idx])
    probe.eval()

    # 4. Inference Loop
    logger.info("Running inference...")
    unified_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pixel_values = batch["pixel_values"]
            labels = batch["label"]
            subject_ids = batch["subject_id"]
            video_paths = batch["video_path"]

            tokens = extract_video_tokens(model=model, pixel_values=pixel_values, device=device)

            logits = probe(tokens)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i in range(len(labels)):
                unified_predictions.append({
                    "subject_id": subject_ids[i],
                    "video_id": video_paths[i],
                    "true_label": int(labels[i].item()),
                    "probs": probs[i].tolist()
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(loader)}")

    # 5. Save Outputs
    output_file = cfg["output_dir"] / "unified_predictions.json"
    with output_file.open("w") as f:
        json.dump(unified_predictions, f, indent=2)

    logger.info(f"Done. Wrote {len(unified_predictions)} predictions to: {output_file}")


if __name__ == "__main__":
    main()