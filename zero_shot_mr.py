import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from open_clip import create_model_and_transforms, tokenize


try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency fallback
    sns = None

from utils import clean_text, read_avi, resolve_video_path, normalize_label, CLASS_ORDER


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--video-path-col", type=str, default="video_filename", help="CSV column for video path"
    )
    parser.add_argument("--label-col", type=str, default="label", help="CSV column for label")
    parser.add_argument(
        "--subject-id-col", type=str, default="patient_id", help="CSV column for subject id"
    )
    parser.add_argument(
        "--template-vocab", type=Path, default=Path("template_vocab.txt"), help="Prompt source"
    )
    parser.add_argument(
        "--model", type=str, default="hf-hub:mkaichristensen/echo-clip", help="OpenCLIP model"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument(
        "--borderline-policy",
        type=str,
        choices=["round_up", "round_down", "drop"],
        default="round_up",
        help="How to map MILD/MODERATE and MODERATE/SEVERE labels",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("zero_shot_mr_outputs"))
    return parser.parse_args()


def extract_mr_prompts(template_vocab_path=None):
    """
    Returns a dictionary of hardcoded, clinically refined, and balanced
    echocardiogram prompts for four classes of mitral regurgitation.

    Args:
        template_vocab_path: Ignored. Kept for signature compatibility with the legacy pipeline.

    Returns:
        dict: A dictionary where keys are classes ('normal', 'mild', 'moderate', 'severe')
              and values are sorted lists of prompt strings.
    """

    prompts = {
        "normal": [
            "There is no mitral regurgitation.",
            "There is trivial to trace mitral valve regurgitation."
        ],
        "mild": [
            "There is mild mitral regurgitation.",
            "There is mild to mild-moderate mitral valve regurgitation."
        ],
        "moderate": [
            "There is moderate mitral regurgitation.",
            "There is moderate to moderate-severe mitral valve regurgitation."
        ],
        "severe": [
            "There is severe mitral regurgitation.",
            "There is severe to very severe mitral valve regurgitation."
        ]
    }

    # Match the original function's return format (lists sorted alphabetically)
    return {k: sorted(v) for k, v in prompts.items()}


def sample_frames(video_np, max_frames=40, frame_step=2):
    frames = video_np[0 : min(max_frames, len(video_np)) : frame_step]
    if len(frames) == 0:
        return video_np[:1]
    return frames


def encode_video(video_path, preprocess_val, model, device, max_frames=40, frame_step=2):
    frames = read_avi(video_path, (224, 224))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    frames = sample_frames(frames, max_frames=max_frames, frame_step=frame_step)
    tensor = torch.stack([preprocess_val(T.ToPILImage()(f)) for f in frames], dim=0)
    tensor = tensor.to(device)
    tensor = tensor.to(torch.bfloat16)

    with torch.no_grad():
        frame_embeddings = F.normalize(model.encode_image(tensor), dim=-1)

    return frame_embeddings


def compute_class_logits(frame_embeddings, class_text_embeddings):
    # Calculate similarities for all frames independently
    similarities = frame_embeddings @ class_text_embeddings.T

    # REMOVED: similarities.mean(dim=0)
    # Returns shape: (num_frames, num_classes)
    return similarities



def collapse_clinical_significant(labels):
    """Collapse labels 0/1 vs 2/3 into a clinical-significant binary target."""
    labels = np.asarray(labels)
    return np.where(np.isin(labels, [2, 3]), 1, 0).astype(int)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, _, preprocess_val = create_model_and_transforms(
        args.model, precision="bf16", device=args.device
    )
    logit_scale = model.logit_scale.exp().item()
    print("Logit scale:", logit_scale)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    prompts_by_class = extract_mr_prompts(args.template_vocab)
    with (args.output_dir / "mr_prompts.json").open("w") as f:
        json.dump(prompts_by_class, f, indent=2)

    print("Prompt counts:")
    for c in CLASS_ORDER:
        print(f"  {c}: {len(prompts_by_class[c])}")

    print("Prompts:")
    for cls_name in CLASS_ORDER:
        for prompt in prompts_by_class[cls_name]:
            print(f"  {cls_name}: {prompt}")

    class_embeddings = []
    with torch.no_grad():
        for cls_name in CLASS_ORDER:
            text_tokens = tokenize(prompts_by_class[cls_name]).to(args.device)
            text_embs = F.normalize(model.encode_text(text_tokens), dim=-1)
            # Average multiple prompt embeddings to form a class prototype.
            class_emb = F.normalize(text_embs.mean(dim=0, keepdim=True), dim=-1)
            class_embeddings.append(class_emb)
    class_embeddings = torch.cat(class_embeddings, dim=0)

    records = []
    dropped = 0
    with args.csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = normalize_label(
                row[args.label_col], borderline_policy=args.borderline_policy
            )
            if label_name is None:
                print(f"Skipping {row} because no label name found in CSV.")
                dropped += 1
                continue

            video_path = resolve_video_path(row[args.video_path_col], args.csv)
            if not video_path.exists():
                print(f"Skipping {row} because no video path found in CSV.")
                dropped += 1
                continue

            records.append(
                {
                    "video_path": str(video_path),
                    "subject_id": str(row[args.subject_id_col]),
                    "label_name": label_name,
                    "label_id": CLASS_ORDER.index(label_name),
                }
            )

    if len(records) == 0:
        raise RuntimeError("No usable rows found in CSV after filtering.")

    print(f"Loaded {len(records)} videos ({dropped} rows skipped).")

    video_rows = []
    subject_logits = defaultdict(list)
    subject_label_ids = {}

    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec['video_path']}")
        frame_embeddings = encode_video(
            rec["video_path"],
            preprocess_val,
            model,
            args.device,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
        )
        # Get frame-level similarities
        frame_similarities = compute_class_logits(frame_embeddings, class_embeddings)

        # Apply the logit scale (from our previous fix!)
        frame_logits = frame_similarities * logit_scale

        # --- TOP-K POOLING ---
        # Let's look at the top 25% of frames (roughly corresponding to systole)
        k = max(1, len(frame_logits) // 4)

        # Extract the highest logit values for each class across the temporal dimension
        topk_logits, _ = torch.topk(frame_logits, k, dim=0)

        # Average only those top frames to get the final video logits
        video_logits = topk_logits.mean(dim=0)
        # ---------------------

        probs = torch.softmax(video_logits, dim=-1)
        pred_id = int(torch.argmax(video_logits).item())

        video_rows.append(
            {
                "video_path": rec["video_path"],
                "subject_id": rec["subject_id"],
                "label": rec["label_name"],
                "pred": CLASS_ORDER[pred_id],
                **{f"logit_{c}": float(video_logits[j].item()) for j, c in enumerate(CLASS_ORDER)},
                **{f"prob_{c}": float(probs[j].item()) for j, c in enumerate(CLASS_ORDER)},
            }
        )
        numpy_logits = video_logits.detach().cpu().to(torch.float32).numpy()
        subject_logits[rec["subject_id"]].append(numpy_logits)
        if rec["subject_id"] in subject_label_ids:
            if subject_label_ids[rec["subject_id"]] != rec["label_id"]:
                raise RuntimeError(
                    f"Subject {rec['subject_id']} has conflicting labels in CSV."
                )
        subject_label_ids[rec["subject_id"]] = rec["label_id"]

    # Aggregate multiple videos for each subject by mean logits.
    subject_rows = []
    for sid, all_logits in subject_logits.items():
        avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
        pred_id = int(np.argmax(avg_logits))
        true_id = int(subject_label_ids[sid])

        subject_rows.append(
            {
                "subject_id": sid,
                "n_videos": len(all_logits),
                "label": CLASS_ORDER[true_id],
                "pred": CLASS_ORDER[pred_id],
                **{f"avg_logit_{c}": float(avg_logits[j]) for j, c in enumerate(CLASS_ORDER)},
            }
        )

    # Build unified predictions (one entry per video) compatible with the unified inference pipeline
    unified_predictions = []
    for vr in video_rows:
        probs = [vr.get(f"prob_{c}", 0.0) for c in CLASS_ORDER]
        try:
            true_label_id = CLASS_ORDER.index(vr["label"]) if vr.get("label") is not None else None
        except ValueError:
            true_label_id = None

        unified_predictions.append(
            {
                "subject_id": vr.get("subject_id"),
                "video_id": vr.get("video_path"),
                "true_label": true_label_id,
                "probs": probs,
            }
        )

    # Save unified predictions for downstream, unified-metrics computation
    unified_out = args.output_dir / "unified_predictions.json"
    with unified_out.open("w") as f:
        json.dump(unified_predictions, f, indent=2)

    # Also save CSV outputs for convenience
    video_out = args.output_dir / "mr_video_predictions.csv"
    with video_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    subject_out = args.output_dir / "mr_subject_predictions.csv"
    with subject_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)

    print("Done.")
    print(f"Wrote unified predictions to: {unified_out}")
    print(f"Wrote prompts to: {args.output_dir / 'mr_prompts.json'}")
    print(f"Wrote video predictions to: {video_out}")
    print(f"Wrote subject predictions to: {subject_out}")


if __name__ == "__main__":
    main()
