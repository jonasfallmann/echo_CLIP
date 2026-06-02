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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency fallback
    sns = None

from utils import clean_text, read_avi


CLASS_ORDER = ["normal", "mild", "moderate", "severe"]
SEVERITY_TERMS = {
    "normal": ["NO", "TRACE", "TRIVIAL"],
    "mild": ["MILD"],
    "moderate": ["MODERATE", "MILD/MODERATE"],
    "severe": ["SEVERE", "MODERATE/SEVERE", "VERY SEVERE"],
}

AMBIGUOUS_PROMPT_FILTER = [
    "CANNOT BE EXCLUDED",
    "HAS IMPROVED",
    "HAS WORSENED",
    "IS UNCHANGED",
    "PARAVALVULAR",
    "RESIDUAL",
    "JET IS",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--video-path-col", type=str, default="video_path", help="CSV column for video path"
    )
    parser.add_argument("--label-col", type=str, default="label", help="CSV column for label")
    parser.add_argument(
        "--subject-id-col", type=str, default="subject_id", help="CSV column for subject id"
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


def normalize_label(label, borderline_policy="round_up"):
    text = clean_text(str(label)).strip()
    if not text:
        return None

    # Support numeric CSV labels where 0=normal, 1=mild, 2=moderate, 3=severe.
    if text in {"0", "1", "2", "3"}:
        return CLASS_ORDER[int(text)]

    text = text.replace(" MITRAL VALVE REGURGITATION", " MITRAL REGURGITATION")
    text = text.replace(" MR", " MITRAL REGURGITATION")

    if "NO MITRAL REGURGITATION" in text or "TRIVIAL MITRAL REGURGITATION" in text:
        return "normal"
    if "TRACE" in text and "MITRAL REGURGITATION" in text:
        return "normal"

    if "MILD/MODERATE" in text:
        if borderline_policy == "round_up":
            return "moderate"
        if borderline_policy == "round_down":
            return "mild"
        return None

    if "MODERATE/SEVERE" in text:
        if borderline_policy == "round_up":
            return "severe"
        if borderline_policy == "round_down":
            return "moderate"
        return None

    if "SEVERE" in text:
        return "severe"
    if "MODERATE" in text:
        return "moderate"
    if "MILD" in text:
        return "mild"

    if text in {"NORMAL", "NONE", "NO", "0"}:
        return "normal"
    return None


def _replace_severity_slots(template, severity_word):
    output = template
    output = re.sub(
        r"<#>(?=\s+TO\s+<#>\s+MITRAL(?:\s+VALVE)?\s+REGURGITATION)",
        severity_word,
        output,
    )
    output = re.sub(
        r"<#>(?=\s+MITRAL(?:\s+VALVE)?\s+REGURGITATION)",
        severity_word,
        output,
    )
    return output


def _fill_numeric_placeholders(prompt, value="50"):
    # Remaining placeholders are typically numeric in these templates.
    return prompt.replace("<#>", value)


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
            "Color flow mapping reveals no evidence of a mitral regurgitation jet in the left atrium.",
            "Doppler findings show a minimal vena contracta, consistent with trace mitral regurgitation.",
            "Doppler findings suggest no mitral regurgitation; the normal left ventricular size is consistent with this finding.",
            "Echocardiogram findings are consistent with a normal mitral valve and no mitral regurgitation.",
            "Echocardiographic assessment indicates trace to trivial mitral regurgitation, which is physiologic.",
            "Normal left ventricular size and absence of volume overload, consistent with chronic trace mitral regurgitation.",
            "Pulmonary vein flow is normal with systolic dominance, consistent with no significant mitral regurgitation.",
            "The mitral valve demonstrates normal leaflet coaptation with no perivalvular or central regurgitation.",
            "There is no mitral regurgitation.",
            "There is trivial to trace mitral valve regurgitation."
        ],
        "mild": [
            "Color flow mapping demonstrates a small central jet, indicating mild mitral regurgitation.",
            "Doppler findings show a narrow vena contracta, consistent with mild mitral regurgitation.",
            "Echocardiogram findings are consistent with mild mitral valve regurgitation.",
            "Normal left ventricular size is observed without evidence of volume overload, consistent with chronic mild mitral regurgitation.",
            "Pulmonary vein flow remains normal with systolic dominance, consistent with mild mitral regurgitation.",
            "The mitral regurgitation jet fills less than 20% of the left atrium, indicating mild mitral regurgitation.",
            "The mitral valve demonstrates mild regurgitation with normal leaflet motion.",
            "The proximal isovelocity surface area (PISA) suggests a small effective regurgitant orifice area, indicating mild mitral regurgitation.",
            "There is mild mitral regurgitation.",
            "There is mild to mild-moderate mitral valve regurgitation."
        ],
        "moderate": [
            "Color flow mapping demonstrates a moderately sized jet, indicating moderate mitral regurgitation.",
            "Doppler findings show an intermediate vena contracta, consistent with moderate mitral regurgitation.",
            "Echocardiogram findings are consistent with moderate mitral valve regurgitation.",
            "Mild left ventricular enlargement and early volume overload are noted, consistent with chronic moderate mitral regurgitation.",
            "The mitral regurgitation jet fills between 20% and 40% of the left atrium, indicating moderate mitral regurgitation.",
            "The mitral valve demonstrates moderate regurgitation due to incomplete leaflet coaptation.",
            "The proximal isovelocity surface area (PISA) derived effective regurgitant orifice area is intermediate, indicating moderate mitral regurgitation.",
            "There is Doppler evidence of blunted systolic forward flow in the pulmonary veins, suggestive of moderate mitral regurgitation.",
            "There is moderate mitral regurgitation.",
            "There is moderate to moderate-severe mitral valve regurgitation."
        ],
        "severe": [
            "Color flow mapping demonstrates a large, sweeping jet, indicating severe mitral regurgitation.",
            "Doppler findings show a wide vena contracta, consistent with severe mitral regurgitation.",
            "Echocardiogram findings are consistent with severe mitral valve regurgitation.",
            "Left ventricular enlargement and significant volume overload are present, consistent with chronic severe mitral regurgitation.",
            "The mitral regurgitation jet fills greater than 50% of the left atrium, indicating severe mitral regurgitation.",
            "The mitral valve demonstrates severe regurgitation with a significant coaptation defect.",
            "The proximal isovelocity surface area (PISA) derived effective regurgitant orifice area is greater than 0.40 cm^2, indicating severe mitral regurgitation.",
            "There is Doppler evidence of systolic flow reversal into the pulmonary veins, indicating severe mitral regurgitation.",
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
    similarities = frame_embeddings @ class_text_embeddings.T
    return similarities.mean(dim=0)


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def collapse_clinical_significant(labels):
    """Collapse labels 0/1 vs 2/3 into a clinical-significant binary target."""
    labels = np.asarray(labels)
    return np.where(np.isin(labels, [2, 3]), 1, 0).astype(int)


def build_metric_tables(y_true, y_pred, class_names, average_modes=None):
    """Build per-class and aggregate metric tables for a task."""
    average_modes = average_modes or ["micro", "macro", "weighted"]
    labels = list(range(len(class_names)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    supports = cm.sum(axis=1)

    per_class_rows = []
    for idx, class_name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)

        per_class_rows.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "sensitivity": recall,
                "f1_score": f1_score,
                "specificity": specificity,
                "support": int(supports[idx]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)

    summary_rows = []
    for avg in average_modes:
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=avg,
            zero_division=0,
        )
        if avg == "micro":
            specificity = np.nan
        elif avg == "macro":
            specificity = float(per_class_df["specificity"].mean()) if not per_class_df.empty else np.nan
        else:
            specificity = float(np.average(per_class_df["specificity"], weights=per_class_df["support"])) if total else np.nan

        summary_rows.append(
            {
                "aggregation": avg,
                "precision": float(precision),
                "recall": float(recall),
                "sensitivity": float(recall),
                "f1_score": float(f1_score),
                "specificity": specificity,
                "accuracy": float(accuracy_score(y_true, y_pred)) if total else np.nan,
                "support": total,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return per_class_df, summary_df, cm


def plot_metric_bars(per_class_df, title, output_path=None):
    """Plot per-class metric bars for precision/recall/F1/specificity."""
    if per_class_df.empty:
        logger.warning("Skipping metric plot for %s because no rows were available.", title)
        return

    plot_df = per_class_df.melt(
        id_vars=["class_name"],
        value_vars=["precision", "recall", "f1_score", "specificity"],
        var_name="metric",
        value_name="value",
    )

    metric_name_map = {
        "precision": "Precision",
        "recall": "Recall / Sensitivity",
        "f1_score": "F1-score",
        "specificity": "Specificity",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_name_map)

    plt.figure(figsize=(max(10, int(len(per_class_df) * 1.4)), 6))
    if sns is not None:
        sns.barplot(data=plot_df, x="class_name", y="value", hue="metric")
        plt.legend(title="Metric", loc="lower right")
    else:
        metrics = ["Precision", "Recall / Sensitivity", "F1-score", "Specificity"]
        metric_to_column = {
            "Precision": "precision",
            "Recall / Sensitivity": "recall",
            "F1-score": "f1_score",
            "Specificity": "specificity",
        }
        classes = list(per_class_df["class_name"])
        x = np.arange(len(classes))
        width = 0.18
        for idx, metric in enumerate(metrics):
            column = metric_to_column[metric]
            values = [float(per_class_df.loc[per_class_df["class_name"] == cls, column].iloc[0]) for cls in classes]
            plt.bar(x + (idx - 1.5) * width, values, width=width, label=metric)
        plt.xticks(x, classes)
        plt.legend(title="Metric", loc="lower right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved metric plot to %s", output_path)

    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None, labels=None):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        ticks = np.arange(len(class_names)) if class_names is not None else np.arange(cm.shape[0])
        plt.xticks(ticks, class_names if class_names is not None else ticks)
        plt.yticks(ticks, class_names if class_names is not None else ticks)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Subject-Level Aggregation)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", output_path)

    plt.close()


def report_metrics(task_name, y_true, y_pred, class_names, output_dir=None):
    """Compute, log, and optionally save metrics for a task."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("Skipping %s report because no predictions were available.", task_name)
        return None

    per_class_df, summary_df, cm = build_metric_tables(y_true, y_pred, class_names)
    labels = list(range(len(class_names)))

    logger.info("\n%s", "=" * 50)
    logger.info("%s", task_name.upper())
    logger.info("%s", "=" * 50)
    logger.info("Accuracy: %.2f%%", accuracy_score(y_true, y_pred) * 100)
    logger.info(
        "\nClassification Report:\n%s",
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        ),
    )
    logger.info("\nPer-class metrics:\n%s", per_class_df.to_string(index=False))
    logger.info("\nAggregate metrics:\n%s", summary_df.to_string(index=False))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = task_name.lower().replace(" ", "_")

        per_class_path = output_dir / f"{prefix}_per_class_metrics.csv"
        summary_path = output_dir / f"{prefix}_summary_metrics.csv"
        plot_path = output_dir / f"{prefix}_metric_bars.png"
        cm_path = output_dir / ("confusion_matrix.png" if prefix == "subject_4way" else f"confusion_matrix_{prefix}.png")

        per_class_df.to_csv(per_class_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved per-class metrics to %s", per_class_path)
        logger.info("Saved summary metrics to %s", summary_path)

        plot_metric_bars(per_class_df, f"{task_name} Metrics", plot_path)
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, output_path=cm_path, labels=labels)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class_df": per_class_df,
        "summary_df": summary_df,
        "confusion_matrix": cm,
    }


def resolve_video_path(path_text, csv_path):
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (csv_path.parent / p).resolve()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, _, preprocess_val = create_model_and_transforms(
        args.model, precision="bf16", device=args.device
    )

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
        logits = compute_class_logits(frame_embeddings, class_embeddings)
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(logits).item())

        video_rows.append(
            {
                "video_path": rec["video_path"],
                "subject_id": rec["subject_id"],
                "label": rec["label_name"],
                "pred": CLASS_ORDER[pred_id],
                **{f"logit_{c}": float(logits[j].item()) for j, c in enumerate(CLASS_ORDER)},
                **{f"prob_{c}": float(probs[j].item()) for j, c in enumerate(CLASS_ORDER)},
            }
        )
        numpy_logits = logits.detach().cpu().to(torch.float32).numpy()
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
