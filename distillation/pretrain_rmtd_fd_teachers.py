#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMTD-FD teacher pretraining script
==================================

Paper-faithful purpose:
    Independently pretrain the three role-specific teachers used before the
    RMTD-FD distillation stage:

    T1 Temporal Context Teacher        -> UniFormer-B, full-frame RGB clips
    T2 Local Action Discrimination     -> UniFormer-S, full-frame RGB clips
    T3 Human-Centered Teacher          -> ROI-UniFormer-B, human ROI clips

Paper settings implemented as defaults:
    - binary fall / non-fall classification
    - 16 sampled frames
    - 224 x 224 input resolution
    - AdamW
    - initial learning rate = 1e-4
    - weight decay = 0.05
    - teacher batch size = 16
    - teacher epochs = 50
    - teacher calibration candidate temperatures = {1, 2, 4}
      and selection by validation negative log-likelihood

Important:
    The paper describes the training protocol but does NOT publish the exact
    local Python module/function names used to build UniFormer-B / UniFormer-S.
    Therefore real training loads each model from an explicit user-supplied
    `module:function` or `/path/file.py:function` factory.

    The paper also does not report an extra LR scheduler, label smoothing,
    MixUp/CutMix, early stopping, or other augmentation recipes. This file does
    not silently add them.

Manifest CSV
------------
Required columns:
    clip_path,roi_path,label

Example:
    clip_path,roi_path,label
    clips/000001.pt,rois/000001.pt,1
    clips/000002.pt,rois/000002.pt,0

For T1/T2, clip_path is used.
For T3, roi_path is used.

Each tensor file may be .pt/.pth/.npy/.npz and should contain one preprocessed
clip in [C,T,H,W], [T,C,H,W], or [T,H,W,C]. By default the script enforces the
paper setting T=16 and H=W=224.

Quick self-test:
    python pretrain_rmtd_fd_teachers.py --smoke-test

Example real commands:
    python pretrain_rmtd_fd_teachers.py \
        --teacher t1 \
        --train-manifest data/train.csv \
        --val-manifest data/val.csv \
        --model-factory models.uniformer:uniformer_base \
        --output-dir runs/t1

    python pretrain_rmtd_fd_teachers.py \
        --teacher t2 \
        --train-manifest data/train.csv \
        --val-manifest data/val.csv \
        --model-factory models.uniformer:uniformer_small \
        --output-dir runs/t2

    python pretrain_rmtd_fd_teachers.py \
        --teacher t3 \
        --train-manifest data/train.csv \
        --val-manifest data/val.csv \
        --model-factory models.roi_uniformer:roi_uniformer_base \
        --output-dir runs/t3

Label convention:
    default positive/fall class index = 1.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
# Tensor / clip loading
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
                    f"{path}: no tensor found under clip/video/frames/tensor/x."
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
            f"{path}: unsupported clip type {path.suffix}; use .pt/.pth/.npy/.npz."
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
        x = x.permute(1, 0, 2, 3)      # [T,C,H,W] -> [C,T,H,W]
    elif x.shape[-1] in (1, 3):
        x = x.permute(3, 0, 1, 2)      # [T,H,W,C] -> [C,T,H,W]
    else:
        raise ValueError(
            f"{path}: cannot infer channel axis from shape {tuple(x.shape)}."
        )

    x = x.contiguous()

    if not x.is_floating_point():
        x = x.float()
        # Operational input conversion only; if uint8-like, convert to [0,1].
        if x.numel() and x.max().item() > 1.5:
            x = x / 255.0
    else:
        x = x.float()

    return x


class TeacherManifestDataset(Dataset):
    """
    Manifest dataset for independent teacher pretraining.

    T1 / T2 -> full-frame `clip_path`
    T3      -> human-centered `roi_path`

    ROI generation itself is intentionally not recreated here because the paper
    specifies that preprocessing separately (YOLOv8n person ROI generation).
    """

    def __init__(
        self,
        manifest: str,
        teacher: str,
        expected_frames: int = 16,
        expected_size: int = 224,
        strict_shape: bool = True,
        allow_missing_roi: bool = False,
    ) -> None:
        self.path = Path(manifest).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.root = self.path.parent
        self.teacher = teacher.lower()
        if self.teacher not in ("t1", "t2", "t3"):
            raise ValueError("teacher must be one of: t1, t2, t3")

        self.expected_frames = int(expected_frames)
        self.expected_size = int(expected_size)
        self.strict_shape = bool(strict_shape)
        self.allow_missing_roi = bool(allow_missing_roi)

        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"Manifest is empty: {self.path}")

        cols = set(rows[0].keys())
        required = {"clip_path", "label"}
        missing = required - cols
        if missing:
            raise ValueError(
                f"Manifest missing required column(s): {sorted(missing)}"
            )

        if self.teacher == "t3" and "roi_path" not in cols and not allow_missing_roi:
            raise ValueError(
                "T3 is the human-centered ROI teacher, but the manifest has no "
                "`roi_path` column. Generate ROI clips first, or explicitly pass "
                "--allow-missing-roi for the paper's detection-failure fallback cases."
            )

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    def _check_shape(self, x: torch.Tensor, path: Path) -> None:
        if not self.strict_shape:
            return
        _, t, h, w = x.shape
        if (
            t != self.expected_frames
            or h != self.expected_size
            or w != self.expected_size
        ):
            raise ValueError(
                f"{path}: got [C,T,H,W]={tuple(x.shape)}, but the paper setting "
                f"is T={self.expected_frames}, H=W={self.expected_size}. "
                f"Preprocess clips accordingly or pass --no-strict-input-shape."
            )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]

        full_path = self._resolve(row["clip_path"])
        if not full_path.exists():
            raise FileNotFoundError(full_path)

        if self.teacher in ("t1", "t2"):
            input_path = full_path
        else:
            roi_raw = (row.get("roi_path") or "").strip()
            if roi_raw:
                roi_path = self._resolve(roi_raw)
                if roi_path.exists():
                    input_path = roi_path
                elif self.allow_missing_roi:
                    input_path = full_path
                else:
                    raise FileNotFoundError(roi_path)
            elif self.allow_missing_roi:
                input_path = full_path
            else:
                raise ValueError(
                    f"Manifest row {index + 2}: empty roi_path for T3."
                )

        clip = _load_tensor_file(input_path)
        self._check_shape(clip, input_path)

        try:
            label = int(row["label"])
        except Exception as exc:
            raise ValueError(
                f"Manifest row {index + 2}: invalid label {row.get('label')!r}"
            ) from exc

        if label not in (0, 1):
            raise ValueError(
                f"Binary fall detection expects label 0/1; got {label} "
                f"at manifest row {index + 2}."
            )

        return {
            "clip": clip,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Model factory / checkpoint utilities
# ---------------------------------------------------------------------

def import_factory(spec: str):
    """
    Supports:
        package.module:function
        /absolute/or/relative/file.py:function
    """
    if ":" not in spec:
        raise ValueError(
            f"Factory must be module:function or file.py:function, got {spec!r}"
        )

    module_spec, function_name = spec.rsplit(":", 1)

    if module_spec.endswith(".py") or os.path.sep in module_spec:
        module_path = Path(module_spec).expanduser().resolve()
        if not module_path.exists():
            raise FileNotFoundError(module_path)

        module_name = f"rmtd_teacher_factory_{abs(hash(str(module_path)))}"
        info = importlib.util.spec_from_file_location(module_name, module_path)
        if info is None or info.loader is None:
            raise ImportError(f"Cannot import {module_path}")

        module = importlib.util.module_from_spec(info)
        sys.modules[module_name] = module
        info.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise AttributeError(
            f"{module_spec!r} has no callable {function_name!r}"
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
                    f"Factory returned {type(model).__name__}, expected nn.Module."
                )
            return model
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not build model with factory {factory_spec!r}. "
        f"Last error: {last_error}"
    )


def unwrap_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in (
            "model",
            "state_dict",
            "model_state_dict",
            "teacher",
            "student",
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


def load_initial_weights(
    model: nn.Module,
    checkpoint: str,
    strict: bool = False,
) -> None:
    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")

    state = unwrap_state_dict(raw)
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


# ---------------------------------------------------------------------
# Generic model forward adapter
# ---------------------------------------------------------------------

def _collect_tensors(obj: Any, out: List[torch.Tensor]) -> None:
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, Mapping):
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
        f"Cannot find logits [B,{num_classes}] for B={batch_size}. "
        f"Returned tensor shapes: {shapes}"
    )


class ModelCaller:
    """
    Supports two common video-model call styles:
        model(clip)
        model([clip])

    `auto` tries list first and then tensor once, then remembers the successful form.
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
        self.input_style = input_style
        self.num_classes = num_classes
        self.resolved: Optional[str] = (
            None if input_style == "auto" else input_style
        )

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        b = clip.shape[0]
        styles = [self.resolved] if self.resolved else ["list", "tensor"]
        errors: List[str] = []

        for style in styles:
            try:
                output = self.model([clip]) if style == "list" else self.model(clip)
                logits = extract_logits(output, b, self.num_classes)

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
            "Model forward failed. Attempts -> " + " | ".join(errors)
        )


# ---------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------

class BinaryMeter:
    def __init__(self, positive_class: int = 1) -> None:
        self.pos = positive_class
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        pred = logits.argmax(dim=1)

        y_pos = labels.eq(self.pos)
        p_pos = pred.eq(self.pos)

        self.tp += int((p_pos & y_pos).sum().item())
        self.fp += int((p_pos & ~y_pos).sum().item())
        self.tn += int((~p_pos & ~y_pos).sum().item())
        self.fn += int((~p_pos & y_pos).sum().item())

    def compute(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.tn + self.fn

        acc = (self.tp + self.tn) / max(total, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)

        denom = precision + recall
        f1 = 0.0 if denom == 0.0 else 2 * precision * recall / denom

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


# ---------------------------------------------------------------------
# Train / validation
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    caller: ModelCaller,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    positive_class: int,
    log_interval: int,
) -> Dict[str, float]:
    model.train()

    meter = BinaryMeter(positive_class)
    total_loss = 0.0
    total_count = 0

    start = time.time()

    for step, batch in enumerate(loader, start=1):
        clip = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = caller(clip)
            loss = F.cross_entropy(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {step}: {loss.item()}"
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = labels.numel()
        total_loss += float(loss.detach().item()) * n
        total_count += n
        meter.update(logits.detach(), labels)

        if (
            log_interval > 0
            and (
                step == 1
                or step % log_interval == 0
                or step == len(loader)
            )
        ):
            elapsed = time.time() - start
            avg = total_loss / max(total_count, 1)
            print(
                f"  step {step:4d}/{len(loader):4d} "
                f"loss={avg:.4f} time={elapsed:.1f}s"
            )

    result = meter.compute()
    result["loss"] = total_loss / max(total_count, 1)
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    caller: ModelCaller,
    loader: DataLoader,
    device: torch.device,
    positive_class: int,
) -> Dict[str, float]:
    model.eval()

    meter = BinaryMeter(positive_class)
    ce_sum = 0.0
    count = 0

    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = caller(clip)

        ce_sum += float(
            F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()
        )
        count += labels.numel()
        meter.update(logits, labels)

    result = meter.compute()
    result["loss"] = ce_sum / max(count, 1)
    return result


@torch.no_grad()
def collect_validation_logits(
    model: nn.Module,
    caller: ModelCaller,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = caller(clip)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_logits:
        raise RuntimeError("Validation loader is empty.")

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def select_calibration_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidates: Sequence[float] = (1.0, 2.0, 4.0),
) -> Tuple[float, Dict[float, float]]:
    """
    Paper: Tc selected from {1, 2, 4} based on validation negative log-likelihood.
    """
    nll_by_t: Dict[float, float] = {}

    for t in candidates:
        if t <= 0:
            raise ValueError("Calibration temperatures must be > 0.")
        nll = F.cross_entropy(logits / float(t), labels).item()
        nll_by_t[float(t)] = float(nll)

    best_t = min(nll_by_t, key=nll_by_t.get)
    return float(best_t), nll_by_t


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    teacher: str,
    val_metrics: Optional[Mapping[str, float]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "teacher": teacher,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "val_metrics": dict(val_metrics or {}),
        "args": vars(args),
    }
    torch.save(payload, path)


def save_weights_only(
    path: Path,
    model: nn.Module,
    epoch: int,
    teacher: str,
    val_metrics: Optional[Mapping[str, float]],
    calibration_temperature: Optional[float] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "teacher": teacher,
        "model": model.state_dict(),
        "val_metrics": dict(val_metrics or {}),
    }
    if calibration_temperature is not None:
        payload["calibration_temperature"] = calibration_temperature
    torch.save(payload, path)


# ---------------------------------------------------------------------
# Tiny self-test model and dataset
# ---------------------------------------------------------------------

class TinyTeacher(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 12, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(12, num_classes)

    def forward(self, x: Any) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.features(x)
        x = x.mean(dim=(2, 3, 4))
        return self.head(x)


def run_smoke_test() -> None:
    print("[Smoke test] RMTD-FD teacher pretraining")

    set_seed(123)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clip_dir = root / "clips"
        roi_dir = root / "rois"
        clip_dir.mkdir()
        roi_dir.mkdir()

        rows = []
        for i in range(8):
            # Small shape only for self-test.
            clip = torch.randn(3, 4, 8, 8)
            roi = torch.randn(3, 4, 8, 8)
            torch.save(clip, clip_dir / f"{i}.pt")
            torch.save(roi, roi_dir / f"{i}.pt")
            rows.append(
                {
                    "clip_path": f"clips/{i}.pt",
                    "roi_path": f"rois/{i}.pt",
                    "label": i % 2,
                }
            )

        manifest = root / "train.csv"
        with manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["clip_path", "roi_path", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)

        for teacher in ("t1", "t2", "t3"):
            ds = TeacherManifestDataset(
                str(manifest),
                teacher=teacher,
                expected_frames=4,
                expected_size=8,
                strict_shape=True,
            )
            loader = DataLoader(
                ds,
                batch_size=4,
                shuffle=True,
                num_workers=0,
            )

            model = TinyTeacher().to(device)
            caller = ModelCaller(model, input_style="auto")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.05,
            )
            scaler = torch.amp.GradScaler("cpu", enabled=False)

            train_result = train_one_epoch(
                model=model,
                caller=caller,
                loader=loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=False,
                positive_class=1,
                log_interval=0,
            )

            val_result = evaluate(
                model=model,
                caller=caller,
                loader=loader,
                device=device,
                positive_class=1,
            )

            logits, labels = collect_validation_logits(
                model,
                caller,
                loader,
                device,
            )

            best_t, nlls = select_calibration_temperature(
                logits,
                labels,
                (1.0, 2.0, 4.0),
            )

            assert np.isfinite(train_result["loss"])
            assert np.isfinite(val_result["loss"])
            assert best_t in (1.0, 2.0, 4.0)

            print(
                f"  {teacher.upper()} PASS | "
                f"train_loss={train_result['loss']:.4f} | "
                f"val_F1={val_result['f1']:.4f} | "
                f"best_Tc={best_t:g}"
            )

    print("[Smoke test] PASS")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent pretraining for RMTD-FD role-specific teachers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a self-contained CPU test and exit.",
    )

    p.add_argument(
        "--teacher",
        choices=("t1", "t2", "t3"),
        default=None,
        help="t1=Temporal UniFormer-B, t2=Local UniFormer-S, t3=ROI-UniFormer-B.",
    )

    p.add_argument("--train-manifest", type=str, default=None)
    p.add_argument("--val-manifest", type=str, default=None)

    p.add_argument(
        "--model-factory",
        type=str,
        default=None,
        help="Model constructor as module:function or file.py:function.",
    )
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional initialization checkpoint.",
    )
    p.add_argument(
        "--strict-checkpoint",
        action="store_true",
    )
    p.add_argument(
        "--input-style",
        choices=("auto", "tensor", "list"),
        default="auto",
    )

    # Paper settings.
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)

    p.add_argument(
        "--calibration-candidates",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0),
        help="Paper candidate set for validation-NLL teacher calibration.",
    )

    p.add_argument(
        "--fall-class-index",
        type=int,
        choices=(0, 1),
        default=1,
    )

    # Data / runtime.
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument(
        "--allow-missing-roi",
        action="store_true",
        help=(
            "For T3 only: fall back to full frame when an ROI is missing. "
            "Use only for intended detection-failure fallback cases."
        ),
    )
    p.add_argument(
        "--no-strict-input-shape",
        action="store_true",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP on CUDA.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument(
        "--output-dir",
        type=str,
        default="./runs/rmtd_fd_teacher",
    )

    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test:
        return

    required = {
        "--teacher": args.teacher,
        "--train-manifest": args.train_manifest,
        "--val-manifest": args.val_manifest,
        "--model-factory": args.model_factory,
    }

    missing = [
        name for name, value in required.items()
        if value is None or value == ""
    ]

    if missing:
        raise SystemExit(
            "Missing required argument(s): "
            + ", ".join(missing)
            + "\nRun --smoke-test first if you only want to verify this file."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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

    teacher_name = {
        "t1": "Temporal Context Teacher / UniFormer-B",
        "t2": "Local Action Discrimination Teacher / UniFormer-S",
        "t3": "Human-Centered Teacher / ROI-UniFormer-B",
    }[args.teacher]

    print("=" * 72)
    print("RMTD-FD independent teacher pretraining")
    print(f"Teacher: {args.teacher.upper()} - {teacher_name}")
    print(f"Device : {device}")
    print(
        "Paper defaults -> "
        f"frames={args.num_frames}, size={args.image_size}, "
        f"batch={args.batch_size}, epochs={args.epochs}, "
        f"AdamW lr={args.lr:g}, weight_decay={args.weight_decay:g}"
    )
    print("=" * 72)

    train_ds = TeacherManifestDataset(
        manifest=args.train_manifest,
        teacher=args.teacher,
        expected_frames=args.num_frames,
        expected_size=args.image_size,
        strict_shape=not args.no_strict_input_shape,
        allow_missing_roi=args.allow_missing_roi,
    )

    val_ds = TeacherManifestDataset(
        manifest=args.val_manifest,
        teacher=args.teacher,
        expected_frames=args.num_frames,
        expected_size=args.image_size,
        strict_shape=not args.no_strict_input_shape,
        allow_missing_roi=args.allow_missing_roi,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
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

    if args.init_checkpoint:
        load_initial_weights(
            model,
            args.init_checkpoint,
            strict=args.strict_checkpoint,
        )

    caller = ModelCaller(
        model,
        input_style=args.input_style,
        num_classes=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=use_amp,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "run_args.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_result = train_one_epoch(
            model=model,
            caller=caller,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            positive_class=args.fall_class_index,
            log_interval=args.log_interval,
        )

        val_result = evaluate(
            model=model,
            caller=caller,
            loader=val_loader,
            device=device,
            positive_class=args.fall_class_index,
        )

        print(
            "Train | "
            f"loss={train_result['loss']:.4f} "
            f"Acc={100*train_result['accuracy']:.2f}% "
            f"P={100*train_result['precision']:.2f}% "
            f"R={100*train_result['recall']:.2f}% "
            f"F1={100*train_result['f1']:.2f}%"
        )

        print(
            "Val   | "
            f"loss={val_result['loss']:.4f} "
            f"Acc={100*val_result['accuracy']:.2f}% "
            f"P={100*val_result['precision']:.2f}% "
            f"R={100*val_result['recall']:.2f}% "
            f"F1={100*val_result['f1']:.2f}%"
        )

        save_checkpoint(
            out_dir / "last_training.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            teacher=args.teacher,
            val_metrics=val_result,
            args=args,
        )

        save_weights_only(
            out_dir / "last_teacher.pt",
            model=model,
            epoch=epoch,
            teacher=args.teacher,
            val_metrics=val_result,
        )

        # Operational convenience only: preserve highest validation-F1 snapshot.
        # This does not add a new loss or training mechanism.
        if val_result["f1"] > best_f1:
            best_f1 = val_result["f1"]
            best_epoch = epoch

            save_weights_only(
                out_dir / "best_val_f1_teacher.pt",
                model=model,
                epoch=epoch,
                teacher=args.teacher,
                val_metrics=val_result,
            )

            print(
                f"Saved best_val_f1_teacher.pt "
                f"(epoch={epoch}, F1={100*best_f1:.2f}%)"
            )

    # Paper-reported calibration procedure: choose Tc from {1,2,4}
    # according to validation negative log-likelihood.
    logits, labels = collect_validation_logits(
        model=model,
        caller=caller,
        loader=val_loader,
        device=device,
    )

    selected_tc, nll_by_t = select_calibration_temperature(
        logits=logits,
        labels=labels,
        candidates=args.calibration_candidates,
    )

    calibration_info = {
        "teacher": args.teacher,
        "selected_temperature": selected_tc,
        "nll_by_temperature": {
            str(k): v for k, v in nll_by_t.items()
        },
        "paper_reported_final_Tc": 2.0,
        "best_validation_f1_epoch": best_epoch,
        "best_validation_f1": best_f1,
    }

    with (out_dir / "calibration.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            calibration_info,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save final teacher with selected calibration metadata.
    save_weights_only(
        out_dir / "final_teacher.pt",
        model=model,
        epoch=args.epochs,
        teacher=args.teacher,
        val_metrics=evaluate(
            model,
            caller,
            val_loader,
            device,
            args.fall_class_index,
        ),
        calibration_temperature=selected_tc,
    )

    print("\nTraining complete.")
    print(f"Output directory: {out_dir}")
    print("Validation NLL by calibration temperature:")
    for t in sorted(nll_by_t):
        print(f"  Tc={t:g}: NLL={nll_by_t[t]:.6f}")
    print(f"Selected Tc on this validation set: {selected_tc:g}")
    print(
        "Paper-reported final teacher calibration temperature: Tc=2.0"
    )
    print(
        "Use the independently pretrained T1/T2/T3 checkpoints in the "
        "subsequent RMTD-FD distillation stage."
    )


if __name__ == "__main__":
    main()
