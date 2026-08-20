#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMTD-FD student test / evaluation
=================================

This file evaluates the final lightweight student after RMTD-FD distillation.

Paper-faithful inference behavior
---------------------------------
1. Inference retains ONLY the UniFormer-XS student.
   The three teachers, ROI branch, and dynamic routing/teacher-weight computation
   are not used during inference.

2. Main reported clip-level metrics:
      Accuracy, Precision, Recall, F1-score
   with the fall class treated as the positive class.

3. Online fall decision protocol:
      - sliding-window fall probability p_t
      - causal moving average over M windows
      - alarm threshold theta
      - require C consecutive smoothed windows >= theta
      - consecutive alarm windows are merged into one event
   Paper defaults:
      M = 3
      C = 2
      theta = 0.65

4. Input protocol inherited from the paper:
      16 sampled frames, input resolution 224x224.
   The preprocessing files generated earlier store tensors as [C,T,H,W].

Important reproducibility notes
-------------------------------
- The manuscript reports the evaluation protocol but does not publish the exact
  local Python module/function names used to construct UniFormer-XS. Therefore
  real evaluation loads the model from:
      module:function
  or:
      /path/to/file.py:function

- The script does NOT add test-time augmentation, ensembling, teacher inference,
  ROI inference, or any other unreported evaluation mechanism.

- The online alarm logic is exported as sequence-level event records. The paper
  defines this protocol but does not report an additional event-level metric
  formula in the provided manuscript; therefore this script does not invent one.

Input manifest
--------------
Compatible with preprocess_upfall_rmtd_fd.py and preprocess_urfall_rmtd_fd.py.

Required:
    clip_path,label

Optional but used for online sequence reconstruction:
    sequence_id,window_id,start_frame,split,dataset

Example:
    clip_path,roi_path,label,dataset,split,sequence_id,window_id,start_frame
    clips/test/a.pt,rois/test/a.pt,1,UP-Fall,test,S01_A01_T01_C1,0,0

Outputs
-------
output_dir/
    evaluation_summary.json
    predictions.csv
    online_alarms.csv          (when sequence_id exists and --online-eval is enabled)
    sequence_summary.csv       (when sequence_id exists and --online-eval is enabled)

Quick self-test:
    python evaluate_rmtd_fd.py --smoke-test

Real example:
    python evaluate_rmtd_fd.py \
        --manifest /data/RMTD_FD/UP_Fall/test.csv \
        --model-factory models.uniformer:uniformer_xs \
        --checkpoint runs/distill/best_student.pt \
        --output-dir runs/eval_upfall \
        --online-eval \
        --amp
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------

def _load_tensor_file(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")

        if torch.is_tensor(obj):
            x = obj
        elif isinstance(obj, Mapping):
            x = None
            for key in ("clip", "video", "frames", "tensor", "x"):
                if key in obj and torch.is_tensor(obj[key]):
                    x = obj[key]
                    break
            if x is None:
                raise ValueError(
                    f"{path}: cannot find a tensor under clip/video/frames/tensor/x."
                )
        else:
            raise TypeError(
                f"{path}: unsupported saved object type {type(obj).__name__}."
            )

    elif suffix == ".npy":
        x = torch.from_numpy(np.load(path))

    elif suffix == ".npz":
        z = np.load(path)
        if not z.files:
            raise ValueError(f"{path}: empty npz file.")
        x = torch.from_numpy(z[z.files[0]])

    else:
        raise ValueError(
            f"{path}: unsupported extension {path.suffix}; "
            "use .pt/.pth/.npy/.npz."
        )

    if x.dim() == 5 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.dim() != 4:
        raise ValueError(
            f"{path}: expected one 4-D clip, got shape {tuple(x.shape)}."
        )

    # Convert common layouts to [C,T,H,W].
    if x.shape[0] in (1, 3):
        pass
    elif x.shape[1] in (1, 3):
        x = x.permute(1, 0, 2, 3)
    elif x.shape[-1] in (1, 3):
        x = x.permute(3, 0, 1, 2)
    else:
        raise ValueError(
            f"{path}: cannot infer channel dimension from {tuple(x.shape)}."
        )

    x = x.contiguous()

    # Keep the same operational conversion used by the earlier generated
    # training/pretraining files: uint8-like data -> float [0,1].
    if not x.is_floating_point():
        x = x.float()
        if x.numel() and x.max().item() > 1.5:
            x = x / 255.0
    else:
        x = x.float()

    return x


class EvaluationManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        expected_frames: int = 16,
        expected_size: int = 224,
        strict_shape: bool = True,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)

        self.root = self.manifest.parent
        self.expected_frames = int(expected_frames)
        self.expected_size = int(expected_size)
        self.strict_shape = bool(strict_shape)

        with self.manifest.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")

        required = {"clip_path", "label"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(
                f"Manifest must contain {sorted(required)}; missing {sorted(missing)}."
            )

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]

        clip_path = self._resolve(row["clip_path"])
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)

        clip = _load_tensor_file(clip_path)

        if self.strict_shape:
            _, t, h, w = clip.shape
            if (
                t != self.expected_frames
                or h != self.expected_size
                or w != self.expected_size
            ):
                raise ValueError(
                    f"{clip_path}: got [C,T,H,W]={tuple(clip.shape)}, "
                    f"expected T={self.expected_frames}, H=W={self.expected_size}. "
                    "Use the paper preprocessing or pass --no-strict-input-shape."
                )

        try:
            label = int(row["label"])
        except Exception as exc:
            raise ValueError(
                f"Manifest row {index + 2}: invalid label {row.get('label')!r}"
            ) from exc

        if label not in (0, 1):
            raise ValueError(
                f"Binary fall detection expects 0/1 labels; got {label} "
                f"at manifest row {index + 2}."
            )

        def optional_int(name: str, default: int = -1) -> int:
            raw = (row.get(name) or "").strip()
            if raw == "":
                return default
            try:
                return int(raw)
            except Exception:
                return default

        return {
            "clip": clip,
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
            "sequence_id": row.get("sequence_id", "") or "",
            "window_id": optional_int("window_id", index),
            "start_frame": optional_int("start_frame", -1),
            "dataset": row.get("dataset", "") or "",
            "split": row.get("split", "") or "",
            "clip_path": row["clip_path"],
        }


# ---------------------------------------------------------------------
# Model factory / checkpoint
# ---------------------------------------------------------------------

def import_factory(spec: str):
    if ":" not in spec:
        raise ValueError(
            f"Factory must be module:function or file.py:function; got {spec!r}."
        )

    module_spec, function_name = spec.rsplit(":", 1)

    if module_spec.endswith(".py") or os.path.sep in module_spec:
        path = Path(module_spec).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        module_name = f"rmtd_eval_factory_{abs(hash(str(path)))}"
        info = importlib.util.spec_from_file_location(module_name, path)
        if info is None or info.loader is None:
            raise ImportError(f"Cannot import module from {path}")

        module = importlib.util.module_from_spec(info)
        sys.modules[module_name] = module
        info.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise AttributeError(
            f"{module_spec!r} has no callable {function_name!r}."
        )
    return factory


def build_model(factory_spec: str, num_classes: int = 2) -> nn.Module:
    factory = import_factory(factory_spec)

    attempts = (
        {"num_classes": num_classes},
        {"n_classes": num_classes},
        {"num_class": num_classes},
        {"classes": num_classes},
        {},
    )

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            model = factory(**kwargs)
            if not isinstance(model, nn.Module):
                raise TypeError(
                    f"Factory returned {type(model).__name__}; expected nn.Module."
                )
            return model
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not call model factory {factory_spec!r}. Last error: {last_error}"
    )


def unwrap_student_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """
    Supports checkpoints generated by the earlier RMTD-FD distillation file:
        {"student": state_dict, ...}
    plus several common formats.
    """
    if isinstance(obj, Mapping):
        for key in (
            "student",
            "model",
            "state_dict",
            "model_state_dict",
            "net",
            "network",
        ):
            value = obj.get(key)
            if isinstance(value, Mapping) and any(
                torch.is_tensor(v) for v in value.values()
            ):
                obj = value
                break

    if not isinstance(obj, Mapping):
        raise TypeError("Checkpoint does not contain a recognizable state_dict.")

    state = {
        k: v for k, v in obj.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }

    if not state:
        raise TypeError("Checkpoint state_dict is empty or unrecognized.")

    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state


def load_student_checkpoint(
    model: nn.Module,
    checkpoint: str,
    strict: bool = True,
) -> Dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")

    state = unwrap_student_state_dict(raw)
    result = model.load_state_dict(state, strict=strict)

    if not strict:
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            print(
                f"[checkpoint] non-strict load: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing[:8]:
                print("  missing examples:", missing[:8])
            if unexpected[:8]:
                print("  unexpected examples:", unexpected[:8])

    metadata: Dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for key in ("epoch", "val_metrics", "paper_hparams", "args"):
            if key in raw:
                metadata[key] = raw[key]

    return metadata


# ---------------------------------------------------------------------
# Generic UniFormer-style forward adapter
# ---------------------------------------------------------------------

def _collect_tensors(obj: Any, out: List[torch.Tensor]) -> None:
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, Mapping):
        # Prioritize common logits keys.
        for key in ("logits", "pred", "prediction", "output"):
            if key in obj:
                _collect_tensors(obj[key], out)
        for value in obj.values():
            _collect_tensors(value, out)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            _collect_tensors(value, out)


def extract_logits(
    output: Any,
    batch_size: int,
    num_classes: int = 2,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    _collect_tensors(output, tensors)

    for x in tensors:
        if (
            x.dim() == 2
            and x.shape[0] == batch_size
            and x.shape[1] == num_classes
        ):
            return x

    shapes = [tuple(x.shape) for x in tensors]
    raise RuntimeError(
        f"Could not find logits [B,{num_classes}] for B={batch_size}. "
        f"Returned tensor shapes: {shapes}"
    )


class ModelCaller:
    """
    Supports:
        model(clip)
        model([clip])

    The first successful style is remembered.
    """

    def __init__(
        self,
        model: nn.Module,
        input_style: str = "auto",
        num_classes: int = 2,
    ) -> None:
        if input_style not in ("auto", "tensor", "list"):
            raise ValueError("input_style must be auto/tensor/list")

        self.model = model
        self.num_classes = num_classes
        self.resolved: Optional[str] = (
            None if input_style == "auto" else input_style
        )

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        b = clip.shape[0]
        styles = [self.resolved] if self.resolved else ["list", "tensor"]
        errors = []

        for style in styles:
            try:
                raw = self.model([clip]) if style == "list" else self.model(clip)
                logits = extract_logits(raw, b, self.num_classes)

                if self.resolved is None:
                    self.resolved = style
                    print(f"[model] auto-detected input style: {style}")
                return logits

            except Exception as exc:
                errors.append(
                    f"{style}: {type(exc).__name__}: {exc}"
                )
                if self.resolved is not None:
                    break

        raise RuntimeError(
            "Student forward failed. Attempts -> " + " | ".join(errors)
        )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def binary_metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, float]:
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    denom = precision + recall
    f1 = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom

    specificity = tn / max(tn + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Additional diagnostic values; the four metrics above are the paper's
        # main reported clip-level metrics.
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total": int(total),
    }


def compute_binary_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    positive_class: int,
) -> Dict[str, float]:
    tp = fp = tn = fn = 0

    for y, p in zip(labels, preds):
        y_pos = int(y) == positive_class
        p_pos = int(p) == positive_class

        if p_pos and y_pos:
            tp += 1
        elif p_pos and not y_pos:
            fp += 1
        elif not p_pos and not y_pos:
            tn += 1
        else:
            fn += 1

    return binary_metrics_from_counts(tp, fp, tn, fn)


# ---------------------------------------------------------------------
# Clip-level evaluation
# ---------------------------------------------------------------------

@torch.inference_mode()
def evaluate_clips(
    model: nn.Module,
    caller: ModelCaller,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    fall_class_index: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()

    labels_all: List[int] = []
    preds_all: List[int] = []
    rows_all: List[Dict[str, Any]] = []

    ce_sum = 0.0
    n_total = 0

    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = caller(clip)

        probs = F.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        fall_probs = probs[:, fall_class_index]

        ce_sum += float(
            F.cross_entropy(
                logits.float(),
                labels,
                reduction="sum",
            ).item()
        )
        n_total += labels.numel()

        b = labels.shape[0]
        for i in range(b):
            y = int(labels[i].item())
            p = int(preds[i].item())
            fall_prob = float(fall_probs[i].item())

            labels_all.append(y)
            preds_all.append(p)

            sequence_id = batch["sequence_id"][i]
            dataset_name = batch["dataset"][i]
            split_name = batch["split"][i]
            clip_path = batch["clip_path"][i]

            rows_all.append({
                "row_index": int(batch["row_index"][i].item()),
                "clip_path": str(clip_path),
                "dataset": str(dataset_name),
                "split": str(split_name),
                "sequence_id": str(sequence_id),
                "window_id": int(batch["window_id"][i].item()),
                "start_frame": int(batch["start_frame"][i].item()),
                "label": y,
                "prediction": p,
                "fall_probability": fall_prob,
                "correct": int(y == p),
            })

    metrics = compute_binary_metrics(
        labels_all,
        preds_all,
        positive_class=fall_class_index,
    )
    metrics["cross_entropy"] = ce_sum / max(n_total, 1)

    return metrics, rows_all


# ---------------------------------------------------------------------
# Online alarm protocol from the paper
# ---------------------------------------------------------------------

def causal_moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("Moving-average window M must be > 0.")

    q: deque = deque()
    total = 0.0
    out = []

    for x in values:
        q.append(float(x))
        total += float(x)

        if len(q) > window:
            total -= q.popleft()

        out.append(total / len(q))

    return out


def build_online_alarm_events(
    prediction_rows: Sequence[Dict[str, Any]],
    moving_average_window: int,
    consecutive_windows: int,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reconstruct the paper's online alarm logic sequence by sequence.

    No event-level Accuracy/F1 is invented here, because the provided manuscript
    defines the alarm generation rule but does not define an event-level metric
    formula for the reported tables.
    """
    if consecutive_windows <= 0:
        raise ValueError("C must be > 0.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("theta must be in [0,1].")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in prediction_rows:
        seq = str(row.get("sequence_id", "") or "")
        if seq:
            grouped[seq].append(dict(row))

    events: List[Dict[str, Any]] = []
    seq_summary: List[Dict[str, Any]] = []

    for seq_id, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda r: (
                int(r.get("window_id", 10**12)),
                int(r.get("start_frame", 10**12)),
                int(r.get("row_index", 10**12)),
            ),
        )

        raw_probs = [float(r["fall_probability"]) for r in rows]
        smooth_probs = causal_moving_average(
            raw_probs,
            moving_average_window,
        )
        high = [p >= threshold for p in smooth_probs]

        streak = 0
        in_event = False
        event_id = 0
        current_event: Optional[Dict[str, Any]] = None

        for i, is_high in enumerate(high):
            if is_high:
                streak += 1
            else:
                streak = 0
                if in_event and current_event is not None:
                    current_event["end_window_id"] = int(rows[i - 1]["window_id"])
                    current_event["end_start_frame"] = int(rows[i - 1]["start_frame"])
                    current_event["num_alarm_windows"] = (
                        int(rows[i - 1]["window_id"])
                        - int(current_event["trigger_window_id"])
                        + 1
                    )
                    events.append(current_event)
                    current_event = None
                    in_event = False

            if (
                not in_event
                and is_high
                and streak >= consecutive_windows
            ):
                event_id += 1
                in_event = True

                trigger_row = rows[i]
                # The alarm becomes valid at the current window, i.e. the C-th
                # consecutive smoothed probability satisfying the threshold.
                current_event = {
                    "sequence_id": seq_id,
                    "event_id": event_id,
                    "trigger_window_id": int(trigger_row["window_id"]),
                    "trigger_start_frame": int(trigger_row["start_frame"]),
                    "trigger_smoothed_probability": float(smooth_probs[i]),
                    "M": int(moving_average_window),
                    "C": int(consecutive_windows),
                    "theta": float(threshold),
                }

        if in_event and current_event is not None:
            last = rows[-1]
            current_event["end_window_id"] = int(last["window_id"])
            current_event["end_start_frame"] = int(last["start_frame"])
            current_event["num_alarm_windows"] = (
                int(last["window_id"])
                - int(current_event["trigger_window_id"])
                + 1
            )
            events.append(current_event)

        # Sequence label is informational only. For preprocessed manifests it is
        # normally constant inside one original sequence.
        seq_labels = [int(r["label"]) for r in rows]
        unique_labels = sorted(set(seq_labels))

        seq_summary.append({
            "sequence_id": seq_id,
            "label_values": "|".join(map(str, unique_labels)),
            "num_windows": len(rows),
            "num_alarm_events": event_id,
            "max_raw_fall_probability": max(raw_probs) if raw_probs else float("nan"),
            "max_smoothed_fall_probability": max(smooth_probs) if smooth_probs else float("nan"),
            "first_alarm_window_id": (
                min(
                    [
                        int(e["trigger_window_id"])
                        for e in events
                        if e["sequence_id"] == seq_id
                    ],
                    default=-1,
                )
            ),
        })

    return events, seq_summary


# ---------------------------------------------------------------------
# Speed / parameter diagnostics
# ---------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return int(total), int(trainable)


@torch.inference_mode()
def benchmark_fps(
    model: nn.Module,
    caller: ModelCaller,
    sample_clip: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Optional runtime measurement on the user's current hardware.

    This is a diagnostic benchmark, not a hard-coded reproduction of the
    paper's reported RTX 4090 FPS.
    """
    model.eval()

    sample_clip = sample_clip.to(device)
    if sample_clip.dim() == 4:
        sample_clip = sample_clip.unsqueeze(0)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(max(warmup, 0)):
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            _ = caller(sample_clip)

    sync()
    start = time.perf_counter()

    for _ in range(max(iterations, 1)):
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            _ = caller(sample_clip)

    sync()
    elapsed = time.perf_counter() - start

    clips = max(iterations, 1) * sample_clip.shape[0]
    fps = clips / max(elapsed, 1e-12)

    return {
        "benchmark_batch_size": int(sample_clip.shape[0]),
        "benchmark_iterations": int(max(iterations, 1)),
        "elapsed_seconds": float(elapsed),
        "clips_per_second": float(fps),
        # With one 16-frame clip as one input unit, this is throughput of clips,
        # not source-video frame rate. The name is explicit to avoid confusion.
    }


# ---------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        # Still create an empty file for deterministic workflows.
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Tiny smoke-test model
# ---------------------------------------------------------------------

class TinyStudent(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 6, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(6, 8, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(8, num_classes)

    def forward(self, x: Any) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.mean(dim=(2, 3, 4))
        return self.head(x)


def run_smoke_test() -> None:
    print("[Smoke test] RMTD-FD test/evaluation")

    set_seed(123)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clips = root / "clips"
        clips.mkdir()

        rows = []

        # Two sequences, four windows each.
        for seq_idx, label in enumerate((0, 1), start=1):
            for w in range(4):
                # Small input for a fast self-test.
                x = torch.randn(3, 4, 8, 8)
                p = clips / f"seq{seq_idx}_w{w}.pt"
                torch.save(x, p)

                rows.append({
                    "clip_path": str(p.relative_to(root)),
                    "label": label,
                    "dataset": "SMOKE",
                    "split": "test",
                    "sequence_id": f"SEQ{seq_idx}",
                    "window_id": w,
                    "start_frame": w * 2,
                })

        manifest = root / "test.csv"
        with manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        ds = EvaluationManifestDataset(
            str(manifest),
            expected_frames=4,
            expected_size=8,
            strict_shape=True,
        )
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        model = TinyStudent().to(device)
        caller = ModelCaller(
            model,
            input_style="auto",
        )

        metrics, preds = evaluate_clips(
            model=model,
            caller=caller,
            loader=loader,
            device=device,
            use_amp=False,
            fall_class_index=1,
        )

        events, seqs = build_online_alarm_events(
            preds,
            moving_average_window=3,
            consecutive_windows=2,
            threshold=0.65,
        )

        assert len(preds) == 8
        assert len(seqs) == 2
        assert math.isfinite(metrics["accuracy"])
        assert math.isfinite(metrics["f1"])

        # Test online alarm rule with a deterministic synthetic probability trace.
        synthetic = []
        for i, prob in enumerate((0.4, 0.8, 0.9, 0.95, 0.2)):
            synthetic.append({
                "row_index": i,
                "sequence_id": "ALARM_TEST",
                "window_id": i,
                "start_frame": i * 8,
                "label": 1,
                "fall_probability": prob,
            })

        alarm_events, _ = build_online_alarm_events(
            synthetic,
            moving_average_window=1,
            consecutive_windows=2,
            threshold=0.65,
        )
        assert len(alarm_events) == 1
        assert alarm_events[0]["trigger_window_id"] == 2

        total_params, _ = count_parameters(model)
        assert total_params > 0

        print(
            "[Smoke test] PASS | "
            f"N={metrics['total']} "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"params={total_params}"
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RMTD-FD final student test/evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--smoke-test", action="store_true")

    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--model-factory", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)

    p.add_argument(
        "--input-style",
        choices=("auto", "tensor", "list"),
        default="auto",
    )
    p.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys.",
    )

    # Paper input settings.
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)

    # Paper positive class / online alarm defaults.
    p.add_argument(
        "--fall-class-index",
        type=int,
        choices=(0, 1),
        default=1,
    )
    p.add_argument(
        "--online-eval",
        action="store_true",
        help="Export online alarm events using sequence_id/window_id.",
    )
    p.add_argument(
        "--moving-average-window",
        type=int,
        default=3,
        help="M in the paper.",
    )
    p.add_argument(
        "--consecutive-windows",
        type=int,
        default=2,
        help="C in the paper.",
    )
    p.add_argument(
        "--alarm-threshold",
        type=float,
        default=0.65,
        help="theta in the paper.",
    )

    # Runtime.
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="./runs/rmtd_fd_eval")
    p.add_argument("--no-strict-input-shape", action="store_true")

    # Optional runtime benchmark.
    p.add_argument("--benchmark-speed", action="store_true")
    p.add_argument("--benchmark-warmup", type=int, default=20)
    p.add_argument("--benchmark-iterations", type=int, default=100)

    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test:
        return

    required = {
        "--manifest": args.manifest,
        "--model-factory": args.model_factory,
        "--checkpoint": args.checkpoint,
    }

    missing = [
        k for k, v in required.items()
        if v is None or v == ""
    ]

    if missing:
        raise SystemExit(
            "Missing required arguments: "
            + ", ".join(missing)
            + "\nRun --smoke-test first if you only want to verify this file."
        )

    if args.moving_average_window <= 0:
        raise SystemExit("--moving-average-window must be > 0.")
    if args.consecutive_windows <= 0:
        raise SystemExit("--consecutive-windows must be > 0.")
    if not (0.0 <= args.alarm_threshold <= 1.0):
        raise SystemExit("--alarm-threshold must be in [0,1].")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested, but torch.cuda.is_available() is False."
        )

    use_amp = bool(args.amp and device.type == "cuda")

    print("=" * 74)
    print("RMTD-FD Student Test / Evaluation")
    print("Inference model: student only (UniFormer-XS)")
    print(f"Device: {device}")
    print(
        f"Paper input: {args.num_frames} frames, "
        f"{args.image_size}x{args.image_size}"
    )
    print(
        f"Online alarm defaults: M={args.moving_average_window}, "
        f"C={args.consecutive_windows}, theta={args.alarm_threshold}"
    )
    print("=" * 74)

    dataset = EvaluationManifestDataset(
        manifest=args.manifest,
        expected_frames=args.num_frames,
        expected_size=args.image_size,
        strict_shape=not args.no_strict_input_shape,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    model = build_model(
        args.model_factory,
        num_classes=2,
    ).to(device)

    ckpt_meta = load_student_checkpoint(
        model,
        args.checkpoint,
        strict=not args.non_strict_checkpoint,
    )

    caller = ModelCaller(
        model,
        input_style=args.input_style,
        num_classes=2,
    )

    total_params, trainable_params = count_parameters(model)

    metrics, prediction_rows = evaluate_clips(
        model=model,
        caller=caller,
        loader=loader,
        device=device,
        use_amp=use_amp,
        fall_class_index=args.fall_class_index,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "predictions.csv",
        prediction_rows,
    )

    print("\nClip-level test results")
    print("-" * 74)
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1-score : {metrics['f1'] * 100:.2f}%")
    print(
        f"Confusion: TP={metrics['tp']} FP={metrics['fp']} "
        f"TN={metrics['tn']} FN={metrics['fn']}"
    )
    print(f"CE loss  : {metrics['cross_entropy']:.6f}")

    online_info: Optional[Dict[str, Any]] = None

    if args.online_eval:
        has_sequence_id = any(
            str(r.get("sequence_id", "") or "")
            for r in prediction_rows
        )

        if not has_sequence_id:
            print(
                "\n[online] sequence_id is absent/empty in the manifest; "
                "online alarm export was skipped."
            )
        else:
            events, sequence_summary = build_online_alarm_events(
                prediction_rows,
                moving_average_window=args.moving_average_window,
                consecutive_windows=args.consecutive_windows,
                threshold=args.alarm_threshold,
            )

            write_csv(
                out_dir / "online_alarms.csv",
                events,
            )
            write_csv(
                out_dir / "sequence_summary.csv",
                sequence_summary,
            )

            online_info = {
                "M": args.moving_average_window,
                "C": args.consecutive_windows,
                "theta": args.alarm_threshold,
                "num_sequences": len(sequence_summary),
                "num_alarm_events": len(events),
                "note": (
                    "Alarm events follow the manuscript protocol. No additional "
                    "event-level Accuracy/F1 formula is invented."
                ),
            }

            print("\nOnline alarm export")
            print("-" * 74)
            print(f"Sequences   : {len(sequence_summary)}")
            print(f"Alarm events: {len(events)}")
            print(
                f"Protocol     : M={args.moving_average_window}, "
                f"C={args.consecutive_windows}, "
                f"theta={args.alarm_threshold}"
            )

    speed_info = None

    if args.benchmark_speed:
        first = dataset[0]["clip"]
        speed_info = benchmark_fps(
            model=model,
            caller=caller,
            sample_clip=first,
            device=device,
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
            use_amp=use_amp,
        )

        print("\nCurrent-hardware runtime benchmark")
        print("-" * 74)
        print(
            f"Throughput: "
            f"{speed_info['clips_per_second']:.2f} clips/s "
            f"(batch={speed_info['benchmark_batch_size']})"
        )
        print(
            "This is measured on the current machine; it is not hard-coded "
            "to the paper's RTX 4090 result."
        )

    summary = {
        "evaluation_file": str(Path(args.manifest).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "student_only_inference": True,
        "num_samples": len(dataset),
        "fall_class_index": args.fall_class_index,
        "paper_input": {
            "num_frames": args.num_frames,
            "image_size": args.image_size,
        },
        "clip_level_metrics": metrics,
        "model": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_parameters_million": total_params / 1e6,
        },
        "checkpoint_metadata": ckpt_meta,
        "online_alarm": online_info,
        "runtime_benchmark": speed_info,
        "args": vars(args),
    }

    with (out_dir / "evaluation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nModel parameters")
    print("-" * 74)
    print(f"Total parameters: {total_params / 1e6:.3f} M")

    print("\nSaved:")
    print(" ", out_dir / "evaluation_summary.json")
    print(" ", out_dir / "predictions.csv")
    if args.online_eval and online_info is not None:
        print(" ", out_dir / "online_alarms.csv")
        print(" ", out_dir / "sequence_summary.csv")


if __name__ == "__main__":
    main()
