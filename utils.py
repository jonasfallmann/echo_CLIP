import torch
import numpy as np
from pathlib import Path
import cv2
import re
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

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

zero_shot_prompts = {
    "ejection_fraction": [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ],
    "pacemaker": [
        "ECHO DENSITY IN RIGHT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
        "ECHO DENSITY IN RIGHT ATRIUM SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
    ],
    "impella": [
        "AN IMPELLA CATHETER IS SEEN AND THE INLET AREA IS 4.0CM FROM THE AORTIC VALVE AND DOES NOT INTERFERE WITH NEIGHBORING STRUCTURES, CONSISTENT WITH CORRECT IMPELLA POSITIONING. THERE IS DENSE TURBULENT COLOR FLOW ABOVE THE AORTIC VALVE, CONSISTENT WITH CORRECT OUTFLOW AREA POSITION ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND IS TOO CLOSE TO OR ENTANGLED IN THE PAPILLARY MUSCLE AND SUBANNULAR STRUCTURES SURROUNDING THE MITRAL VALVE; REPOSITIONING RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN, HOWEVER THE INLET AREA APPEARS TO BE IN THE AORTA OR NEAR THE AORTIC VALVE; REPOSITIONING IS RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND EXTENDS TOO FAR INTO THE LEFT VENTRICLE; REPOSITIONING RECOMMENDED ",
    ],
    "normal_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA SHOWS A NORMAL RESPIRATORY COLLAPSE CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (3MMHG). ",
    ],
    "elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES LESS THAN 50% COLLAPSE CONSISTENT WITH ELEVATED RIGHT ATRIAL PRESSURE (8MMHG). ",
    ],
    "significantly_elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES NO INSPIRATORY COLLAPSE, CONSISTENT WITH SIGNIFICANTLY ELEVATED RIGHT ATRIAL PRESSURE (>15MMHG). ",
    ],
    "pulmonary_artery_pressure": [
        "ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG. ",
        "ESTIMATED PA PRESSURE IS <#>MMHG. ",
        "PA PEAK PRESSURE IS <#>MMHG. ",
    ],
    "severe_left_ventricle_dilation": [
        "SEVERE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "SEVERE DILATED LEFT VENTRICLE BY VOLUME. ",
        "SEVERE DILATED LEFT VENTRICLE. ",
    ],
    "moderate_left_ventricle_dilation": [
        "MODERATE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MODERATE DILATED LEFT VENTRICLE BY VOLUME. ",
        "MODERATE DILATED LEFT VENTRICLE. ",
    ],
    "mild_left_ventricle_dilation": [
        "MILD DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MILD DILATED LEFT VENTRICLE BY VOLUME. ",
        "MILD DILATED LEFT VENTRICLE. ",
    ],
    "severe_right_ventricle_size": ["SEVERE DILATED RIGHT VENTRICLE. "],
    "moderate_right_ventricle_size": ["MODERATE DILATED RIGHT VENTRICLE. "],
    "mild_right_ventricle_size": ["MILD DILATED RIGHT VENTRICLE. "],
    "severe_left_atrium_size": ["SEVERE DILATED LEFT ATRIUM. "],
    "moderate_left_atrium_size": ["MODERATE DILATED LEFT ATRIUM. "],
    "mild_left_atrium_size": ["MILD DILATED LEFT ATRIUM. "],
    "severe_right_atrium_size": ["SEVERE DILATED RIGHT ATRIUM. "],
    "moderate_right_atrium_size": ["MODERATE DILATED RIGHT ATRIUM. "],
    "mild_right_atrium_size": ["MILD DILATED RIGHT ATRIUM. "],
    "tavr": [
        "A BIOPROSTHETIC STENT-VALVE IS PRESENT IN THE AORTIC POSITION. ",
    ],
    "mitraclip": [
        "TWO MITRACLIPS ARE SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
        "TWO MITRACLIPS ARE NOW PRESENT ON THE ANTERIOR AND POSTERIOR MITRAL VALVE LEAFLETS. ",
        "ONE MITRACLIP IS SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
    ],
}


def compute_binary_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
):
    per_frame_similarities = video_embeddings @ prompt_embeddings.T
    # Average along the candidate dimension and frame dimension
    predictions = per_frame_similarities.mean(dim=-1).mean(dim=-1)

    return predictions


def compute_regression_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    per_frame_similarities = (
        video_embeddings @ prompt_embeddings.T
    )  # (N x Frames x Candidates)

    # Sort the candidates by their similarity to the video
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    # Convert matrix of indices to their corresponding continuous values.
    prompt_values = torch.tensor(
        prompt_values, device=video_embeddings.device
    )  # (N x Frames x Candidates)
    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]

    # Taking the mean along dim=1 collapses the frames dimension
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(
        dim=1
    )  # (N x Candidates)

    # The median of only the top 20% of predicted values is taken
    # as the final predicted value
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]

    return final_prediction


def crop_and_scale(img, res=(640, 480), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)

    return img


def read_avi(p: Path, res=None):
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


## TEXT CLEANING UTILS

removables = re.compile(r"\^|CRLF|‡")

in_text_periods = re.compile(r"(?<=\D)\.|\.(?=\D)")
square_brackets = re.compile(r"[\[\]]")
multi_whitespace = re.compile(r"\s+")
multi_period = re.compile(r"\.+")

select_was = re.compile(r"(?<=\b)WAS(?=\b)")
select_were = re.compile(r"(?<=\b)WERE(?=\b)")
select_and_or = re.compile(r"(?<=\b)AND/OR(?=\b)")
select_normally = re.compile(r"NORMALLY")
select_mildly = re.compile(r"MILDLY")
select_moderately = re.compile(r"MODERATELY")
select_severely = re.compile(r"SEVERELY")
select_pa = re.compile(r"PULMONARY ARTERY")
select_icd_codes = re.compile(r"[A-Z](\d+\.\d*\b)")
select_slash_dates = re.compile(r"\d{2}/\d{2}/\d{4}")
select_dot_dates = re.compile(r"\d{2}\.\d{2}\.\d{4}")

space_before_unit = re.compile(r"\s+(MMHG|MM|CM|%)")
space_period = re.compile(r"\s\.")

space_plus_space = re.compile(r"\s\+\s")
verbose_pressure = re.compile(r"\+CVPMMHG")
add_period = [
    r"THE PEAK TRANSAORTIC GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSAORTIC GRADIENT IS <#>MMHG",
    r"LV EJECTION FRACTION IS <#>%",
    r"ESTIMATED PA PRESSURE IS <#>MMHG",
    r"RESTING SEGMENTAL WALL MOTION ANALYSIS",
    r"THE IVC DIAMETER IS <#>MM",
    r"EST RV/RA PRESSURE GRADIENT IS <#>MMHG",
    r"ESTIMATED PEAK RVSP IS <#>MMHG",
    r"HEART FAILURE, UNSPECIFIED",
    r"CHEST PAIN, UNSPECIFIED",
    r"SINUS OF VALSALVA: <#>CM",
    r"THE PEAK TRANSMITRAL GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSMITRAL GRADIENT IS <#>MMHG",
    r"ASCENDING AORTA <#>CM",
    r"ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG",
    r"ICD_CODE SHORTNESS BREATH",
    r"ICD_CODE ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"SHORTNESS BREATH",
    r"ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"THE LEFT ATRIAL APPENDAGE IS NORMAL IN APPEARANCE WITH NO EVIDENCE OF THROMBUS",
]

select_number = r"(?:\d+\.?\d*)"

add_period = [re.escape(a).replace(re.escape("<#>"), select_number) for a in add_period]
add_period = [f"(?:{a})(?!\.)" for a in add_period]
add_period = "|".join(add_period)
add_period = f"({add_period})"
# print(f"{add_period[:50]} ... {add_period[-50:]}")
add_period = re.compile(add_period)


def clean_text(text):
    if len(text) > 1:
        text = text.upper()
        text = text.strip()
        text = text.replace("`", "'")
        text = removables.sub("", text)

        text = in_text_periods.sub(". ", text)
        text = square_brackets.sub("", text)

        text = select_was.sub("IS", text)
        text = select_were.sub("ARE", text)
        text = select_and_or.sub("AND", text)
        text = select_normally.sub("NORMAL", text)
        text = select_mildly.sub("MILD", text)
        text = select_moderately.sub("MODERATE", text)
        text = select_severely.sub("SEVERE", text)
        text = select_pa.sub("PA", text)
        text = select_slash_dates.sub("", text)
        text = select_dot_dates.sub("", text)
        text = select_icd_codes.sub("", text)

        text = space_before_unit.sub(r"\1", text)
        text = space_period.sub(".", text)
        text = multi_whitespace.sub(" ", text)

        text = space_plus_space.sub("+", text)
        text = verbose_pressure.sub("MMHG", text)

        text = text.strip()
        text = text + " "

        text = add_period.sub(r"\1.", text)
        text = multi_period.sub(".", text)

    return text


select_severity = "|".join(
    ["MODERATE/SEVERE", "MILD/MODERATE", "MILD", "MODERATE", "SEVERE", "VERY SEVERE"]
)
select_severity = f"((?<![A-Za-z])(?:{select_severity}))"
select_number = r"(\d+\.?\d*)"

select_variable = "|".join([select_number, select_severity])
# print(select_variable)
select_variable = re.compile(select_variable)


def extract_variables(string, replace_with="<#>"):
    matches = select_variable.findall(string)
    variables = []
    for match in matches:
        for variable in match:
            if not len(variable) == 0:
                variables.append(variable)
    variables_replaced = select_variable.sub(replace_with, string)
    return variables, variables_replaced




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


def resolve_video_path(path_text, csv_path):
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (csv_path.parent / p).resolve()



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


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


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