import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from open_clip import create_model_and_transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attentive_pooler import AttentiveClassifier
from linear_pooler import LinearClassifier, MLPClassifier
from utils import read_avi
from utils import CLASS_ORDER, normalize_label, resolve_video_path, report_metrics

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- SCHEDULERS ---

class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr", 0.0)
            start_lr = group.get("mc_start_lr", 0.0)
            warmup_steps = group.get("mc_warmup_steps", 0)

            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd", ref_wd)

            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd


# --- CORE LOGIC ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train attentive probe(s) on frozen Echo-CLIP video embeddings.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config for multi-head grid search")

    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--video-path-col", type=str, default="video_path")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--subject-id-col", type=str, default="subject_id")
    parser.add_argument("--borderline-policy", type=str, choices=["round_up", "round_down", "drop"], default="round_up")

    parser.add_argument("--model", type=str, default="hf-hub:mkaichristensen/echo-clip")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--frame-step", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use-amp", action="store_true", help="Use Automatic Mixed Precision")

    parser.add_argument("--probe-depth", type=int, default=1)
    parser.add_argument("--probe-num-heads", type=int, default=8)
    parser.add_argument("--probe-mlp-ratio", type=float, default=4.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--output-dir", type=Path, default=Path("probe_mr_outputs"))

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="miracle-echo-clip-probe")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_training_config(args):
    if args.config is None:
        if args.train_csv is None or args.val_csv is None:
            raise ValueError("Provide --config, or both --train-csv and --val-csv.")

        return {
            "seed": args.seed,
            "model": {
                "name": args.model,
                "device": args.device,
            },
            "data": {
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "video_path_col": args.video_path_col,
                "label_col": args.label_col,
                "subject_id_col": args.subject_id_col,
                "borderline_policy": args.borderline_policy,
                "max_frames": args.max_frames,
                "frame_step": args.frame_step,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            "probe": {
                "depth": args.probe_depth,
                "num_heads": args.probe_num_heads,
                "mlp_ratio": args.probe_mlp_ratio,
            },
            "optimization": {
                "num_epochs": args.epochs,
                "use_amp": args.use_amp,
                "multihead_kwargs": [
                    {
                        "name": "head_0",
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "warmup_epochs": 2.0,
                        "start_lr": args.lr * 0.1,
                        "final_lr": args.lr * 0.01,
                        "final_weight_decay": args.weight_decay
                    }
                ],
            },
            "logging": {
                "output_dir": str(args.output_dir),
                "use_wandb": args.use_wandb,
                "wandb_project": args.wandb_project,
                "wandb_run_name": args.wandb_run_name,
            },
        }

    with args.config.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML config root must be a mapping.")
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg):
    if device_arg == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


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


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def extract_video_tokens(model, pixel_values, device):
    bsz, nframes = pixel_values.shape[:2]
    flat_pixels = pixel_values.reshape(-1, *pixel_values.shape[2:])

    model_dtype = next(model.parameters()).dtype
    flat_pixels = flat_pixels.to(device=device, dtype=model_dtype, non_blocking=True)

    with torch.no_grad():
        frame_embeddings = model.encode_image(flat_pixels)
        frame_embeddings = F.normalize(frame_embeddings.float(), dim=-1)

    return frame_embeddings.reshape(bsz, nframes, -1)


def compute_subject_accuracy_per_head(video_probs_by_head, video_subject_ids, video_targets):
    grouped = defaultdict(list)
    target_by_subject = {}

    for i, sid in enumerate(video_subject_ids):
        grouped[sid].append(i)
        target = int(video_targets[i])
        if sid in target_by_subject and target_by_subject[sid] != target:
            raise RuntimeError(f"Subject {sid} has conflicting labels in validation split.")
        target_by_subject[sid] = target

    if not grouped:
        return [0.0 for _ in video_probs_by_head], [[] for _ in video_probs_by_head], [[] for _ in video_probs_by_head]

    num_heads = len(video_probs_by_head)
    per_head_acc = []
    per_head_y_true = []
    per_head_y_pred = []

    for head_idx in range(num_heads):
        y_true = []
        y_pred = []
        probs_for_head = video_probs_by_head[head_idx]

        for sid, indices in grouped.items():
            avg_prob = np.mean(np.stack([probs_for_head[i] for i in indices], axis=0), axis=0)
            y_true.append(int(target_by_subject[sid]))
            y_pred.append(int(np.argmax(avg_prob)))

        acc = 100.0 * float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
        per_head_acc.append(acc)
        per_head_y_true.append(y_true)
        per_head_y_pred.append(y_pred)

    return per_head_acc, per_head_y_true, per_head_y_pred


def make_probe(
    probe_type: str,
    embed_dim: int,
    num_classes: int,
    num_heads: int,
    depth: int,
    mlp_ratio: float,
    dropout: float,
    use_layernorm: bool,
):
    """Factory for probe heads: supports 'attentive', 'linear', and 'mlp'."""
    if probe_type == "linear":
        return LinearClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            use_layernorm=use_layernorm,
            dropout=dropout,
        )
    if probe_type == "mlp":
        return MLPClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            use_layernorm=use_layernorm,
            dropout=dropout,
        )
    # default: attentive
    return AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        use_activation_checkpointing=False,
    )


def init_opt(probes, opt_kwargs, iterations_per_epoch, num_epochs, use_amp):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    total_steps = int(num_epochs * iterations_per_epoch)

    for p, kwargs in zip(probes, opt_kwargs):
        warmup_epochs = kwargs.get("warmup_epochs", 0)
        param_groups = [
            {
                "params": p.parameters(),
                "mc_warmup_steps": int(warmup_epochs * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr", kwargs.get("lr", 1e-4) * 0.1),
                "mc_ref_lr": kwargs.get("lr", 1e-4),
                "mc_final_lr": kwargs.get("final_lr", kwargs.get("lr", 1e-4) * 0.01),
                "mc_ref_wd": kwargs.get("weight_decay", 1e-2),
                "mc_final_wd": kwargs.get("final_weight_decay", kwargs.get("weight_decay", 1e-2)),
            }
        ]

        opt = torch.optim.AdamW(param_groups)
        optimizers.append(opt)
        schedulers.append(WarmupCosineLRSchedule(opt, T_max=total_steps))
        wd_schedulers.append(CosineWDSchedule(opt, T_max=total_steps))
        scalers.append(torch.cuda.amp.GradScaler(enabled=use_amp))

    return optimizers, schedulers, wd_schedulers, scalers


def run_epoch(
        *,
        model,
        probes,
        loader,
        optimizers,
        schedulers,
        wd_schedulers,
        scalers,
        criterion,
        device,
        use_amp,
        training,
):
    for p in probes:
        p.train(mode=training)

    num_heads = len(probes)

    total_loss = [0.0 for _ in range(num_heads)]
    total = 0
    correct = [0 for _ in range(num_heads)]

    video_probs_by_head = [[] for _ in range(num_heads)]
    video_subject_ids = []
    video_targets = []

    for batch in loader:
        if training:
            [s.step() for s in schedulers]
            [wds.step() for wds in wd_schedulers]

        labels = batch["label"].to(device, non_blocking=True)
        tokens = extract_video_tokens(model=model, pixel_values=batch["pixel_values"], device=device)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits_per_head = [probe(tokens) for probe in probes]
            losses = [criterion(logits, labels) for logits in logits_per_head]

        if training:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            for scaler, loss in zip(scalers, losses):
                scaler.scale(loss).backward()

            for scaler, opt in zip(scalers, optimizers):
                scaler.step(opt)
                scaler.update()

        with torch.no_grad():
            for head_idx, logits in enumerate(logits_per_head):
                probs = torch.softmax(logits.detach().float(), dim=-1)
                preds = torch.argmax(probs, dim=-1)

                total_loss[head_idx] += float(losses[head_idx].item()) * labels.size(0)
                correct[head_idx] += int((preds == labels).sum().item())

                if not training:
                    video_probs_by_head[head_idx].extend(probs.cpu().numpy())

        if not training:
            video_subject_ids.extend(batch["subject_id"])
            video_targets.extend(labels.cpu().numpy().tolist())

        total += int(labels.size(0))

    avg_loss = [v / max(1, total) for v in total_loss]
    video_acc = [100.0 * c / max(1, total) for c in correct]

    metrics = {
        "loss_per_head": avg_loss,
        "video_acc_per_head": video_acc,
    }

    if not training:
        subj_acc, subj_y_true, subj_y_pred = compute_subject_accuracy_per_head(
            video_probs_by_head=video_probs_by_head,
            video_subject_ids=video_subject_ids,
            video_targets=video_targets,
        )
        metrics["subject_acc_per_head"] = subj_acc
        metrics["subject_y_true_per_head"] = subj_y_true
        metrics["subject_y_pred_per_head"] = subj_y_pred

    return metrics


def build_head_name(idx, kwargs):
    if kwargs.get("name"):
        return str(kwargs["name"])
    lr = kwargs.get("lr", "na")
    wd = kwargs.get("weight_decay", "na")
    return f"head_{idx}_lr{lr}_wd{wd}"


def save_probe_checkpoint(
        *,
        output_dir,
        epoch,
        probes,
        optimizers,
        scalers,
        head_names,
        best_subject_acc_per_head,
        val_subject_acc_per_head,
        best_head_idx,
        is_best,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "head_names": head_names,
        "probe_state_dicts": [p.state_dict() for p in probes],
        "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        "scaler_state_dicts": [s.state_dict() for s in scalers],
        "best_subject_acc_per_head": [float(v) for v in best_subject_acc_per_head],
        "val_subject_acc_per_head": [float(v) for v in val_subject_acc_per_head],
        "best_head_idx": int(best_head_idx),
    }

    torch.save(state, output_dir / "latest.pt")
    torch.save(state, output_dir / f"epoch_{epoch:03d}.pt")

    if is_best:
        torch.save(state, output_dir / "best.pt")


def maybe_init_wandb(cfg):
    logging_cfg = cfg.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    if wandb is None:
        logger.warning("wandb logging requested but wandb is not installed. Continuing without wandb.")
        return None

    run = wandb.init(
        project=logging_cfg.get("wandb_project", "miracle-echo-clip-probe"),
        name=logging_cfg.get("wandb_run_name"),
        config=cfg,
    )
    return run


def train_probe(cfg):
    set_seed(int(cfg.get("seed", 0)))

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    probe_cfg = cfg.get("probe", {})
    opt_cfg = cfg.get("optimization", {})
    logging_cfg = cfg.get("logging", {})

    device = pick_device(model_cfg.get("device", "cuda"))
    output_dir = Path(logging_cfg.get("output_dir", "probe_mr_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(data_cfg["train_csv"])
    val_csv = Path(data_cfg["val_csv"])

    max_frames = int(data_cfg.get("max_frames", 40))
    frame_step = int(data_cfg.get("frame_step", 2))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))

    probe_type = str(probe_cfg.get("type", "attentive"))
    probe_depth = int(probe_cfg.get("depth", 1))
    probe_num_heads = int(probe_cfg.get("num_heads", 8))
    probe_mlp_ratio = float(probe_cfg.get("mlp_ratio", 4.0))
    probe_dropout = float(probe_cfg.get("dropout", 0.0))
    probe_use_layernorm = bool(probe_cfg.get("use_layernorm", True))

    num_epochs = int(opt_cfg.get("num_epochs", 20))
    use_amp = opt_cfg.get("use_amp", False)
    multihead_kwargs = opt_cfg.get("multihead_kwargs", [])
    if not multihead_kwargs:
        raise ValueError("optimization.multihead_kwargs must contain at least one head config.")

    logger.info("Loading OpenCLIP model: %s", model_cfg.get("name", "hf-hub:mkaichristensen/echo-clip"))
    precision = "bf16" if device.type == "cuda" else "fp32"
    model, _, preprocess_val = create_model_and_transforms(model_cfg.get("name", "hf-hub:mkaichristensen/echo-clip"),
                                                           precision=precision, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device).to(dtype=torch.bfloat16)
        embed_dim = int(model.encode_image(dummy).shape[-1])

    logger.info("Encoder embedding dim: %d", embed_dim)

    train_records, dropped_train = load_records(
        csv_path=train_csv,
        video_path_col=data_cfg.get("video_path_col", "video_path"),
        label_col=data_cfg.get("label_col", "label"),
        subject_id_col=data_cfg.get("subject_id_col", "subject_id"),
        borderline_policy=data_cfg.get("borderline_policy", "round_up"),
    )
    val_records, dropped_val = load_records(
        csv_path=val_csv,
        video_path_col=data_cfg.get("video_path_col", "video_path"),
        label_col=data_cfg.get("label_col", "label"),
        subject_id_col=data_cfg.get("subject_id_col", "subject_id"),
        borderline_policy=data_cfg.get("borderline_policy", "round_up"),
    )

    if not train_records:
        raise RuntimeError("No usable rows in train CSV after filtering.")
    if not val_records:
        raise RuntimeError("No usable rows in val CSV after filtering.")

    logger.info("Train videos: %d (dropped: %d)", len(train_records), dropped_train)
    logger.info("Val videos: %d (dropped: %d)", len(val_records), dropped_val)

    train_ds = EchoClipProbeDataset(
        records=train_records,
        preprocess=preprocess_val,
        max_frames=max_frames,
        frame_step=frame_step,
    )
    val_ds = EchoClipProbeDataset(
        records=val_records,
        preprocess=preprocess_val,
        max_frames=max_frames,
        frame_step=frame_step,
    )

    train_loader = make_loader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    probes = []
    head_names = []

    logger.info("Probe type: %s | dropout: %.2f | use_layernorm: %s", probe_type, probe_dropout, probe_use_layernorm)

    for idx, kwargs in enumerate(multihead_kwargs):
        probe = make_probe(
            probe_type=probe_type,
            embed_dim=embed_dim,
            num_classes=len(CLASS_ORDER),
            num_heads=probe_num_heads,
            depth=probe_depth,
            mlp_ratio=probe_mlp_ratio,
            dropout=probe_dropout,
            use_layernorm=probe_use_layernorm,
        ).to(device)
        probes.append(probe)
        head_names.append(build_head_name(idx, kwargs))

    logger.info("Initialized %d probe heads", len(probes))

    # Log parameter counts for each classifier
    for idx, (probe, head_name) in enumerate(zip(probes, head_names)):
        total_params = sum(p.numel() for p in probe.parameters())
        trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        logger.info(
            "  [%s] %s: total=%,d  trainable=%,d  non-trainable=%,d",
            idx, head_name, total_params, trainable_params, non_trainable_params,
        )

    # Initialize optimizers, schedulers, and scalers
    optimizers, schedulers, wd_schedulers, scalers = init_opt(
        probes=probes,
        opt_kwargs=multihead_kwargs,
        iterations_per_epoch=len(train_loader),
        num_epochs=num_epochs,
        use_amp=use_amp
    )

    criterion = nn.CrossEntropyLoss()
    run = maybe_init_wandb(cfg)

    best_subject_acc_per_head = np.full(len(probes), -np.inf, dtype=float)
    best_global_subject_acc = float("-inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            probes=probes,
            loader=train_loader,
            optimizers=optimizers,
            schedulers=schedulers,
            wd_schedulers=wd_schedulers,
            scalers=scalers,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            training=True,
        )

        val_metrics = run_epoch(
            model=model,
            probes=probes,
            loader=val_loader,
            optimizers=optimizers,
            schedulers=schedulers,
            wd_schedulers=wd_schedulers,
            scalers=scalers,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            training=False,
        )

        val_subject_acc = np.asarray(val_metrics["subject_acc_per_head"], dtype=float)
        best_subject_acc_per_head = np.maximum(best_subject_acc_per_head, val_subject_acc)

        best_head_idx = int(np.argmax(val_subject_acc))
        current_global_subject_acc = float(val_subject_acc[best_head_idx])
        is_best = current_global_subject_acc >= best_global_subject_acc
        if is_best:
            best_global_subject_acc = current_global_subject_acc

        save_probe_checkpoint(
            output_dir=output_dir / "checkpoints",
            epoch=epoch,
            probes=probes,
            optimizers=optimizers,
            scalers=scalers,
            head_names=head_names,
            best_subject_acc_per_head=best_subject_acc_per_head,
            val_subject_acc_per_head=val_subject_acc,
            best_head_idx=best_head_idx,
            is_best=is_best,
        )

        epoch_log = {
            "epoch": epoch,
            "best_head_idx": best_head_idx,
            "best_head_name": head_names[best_head_idx],
            "val_subject_acc_best_head": current_global_subject_acc,
            "best_global_subject_acc": best_global_subject_acc,
        }

        for i, head_name in enumerate(head_names):
            epoch_log[f"{head_name}/train_loss"] = float(train_metrics["loss_per_head"][i])
            epoch_log[f"{head_name}/train_video_acc"] = float(train_metrics["video_acc_per_head"][i])
            epoch_log[f"{head_name}/val_loss"] = float(val_metrics["loss_per_head"][i])
            epoch_log[f"{head_name}/val_video_acc"] = float(val_metrics["video_acc_per_head"][i])
            epoch_log[f"{head_name}/val_subject_acc"] = float(val_metrics["subject_acc_per_head"][i])
            epoch_log[f"{head_name}/best_subject_acc"] = float(best_subject_acc_per_head[i])

            # Log current learning rate and weight decay
            epoch_log[f"{head_name}/lr"] = optimizers[i].param_groups[0]["lr"]
            epoch_log[f"{head_name}/weight_decay"] = optimizers[i].param_groups[0]["weight_decay"]

        history.append(epoch_log)

        logger.info(
            "[Epoch %03d/%03d] best_head=%s val_subject_acc=%.2f%% global_best=%.2f%%",
            epoch,
            num_epochs,
            head_names[best_head_idx],
            current_global_subject_acc,
            best_global_subject_acc,
        )

        if run is not None:
            wandb.log(epoch_log)

    final_val = run_epoch(
        model=model,
        probes=probes,
        loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        wd_schedulers=wd_schedulers,
        scalers=scalers,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        training=False,
    )

    final_subj_acc = np.asarray(final_val["subject_acc_per_head"], dtype=float)
    final_best_idx = int(np.argmax(final_subj_acc))

    report_metrics(
        task_name="Probe Subject 4way",
        y_true=final_val["subject_y_true_per_head"][final_best_idx],
        y_pred=final_val["subject_y_pred_per_head"][final_best_idx],
        class_names=CLASS_ORDER,
        output_dir=output_dir,
    )

    summary = {
        "head_names": head_names,
        "final_subject_acc_per_head": [float(v) for v in final_subj_acc],
        "best_subject_acc_per_head": [float(v) for v in best_subject_acc_per_head],
        "final_best_head_idx": final_best_idx,
        "final_best_head_name": head_names[final_best_idx],
    }

    with (output_dir / "probe_head_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "probe_training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Outputs written to %s", output_dir)

    if run is not None:
        run.finish()


def main():
    args = parse_args()
    cfg = load_training_config(args)
    train_probe(cfg)


if __name__ == "__main__":
    main()