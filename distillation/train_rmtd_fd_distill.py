#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMTD-FD distillation training file (paper-faithful core implementation)

This file implements the training objective described in:
"RMTD-FD: Role-Aware Multi-Teacher Distillation for Lightweight Video-Based Fall Detection"

Implemented supervision:
  T1 Temporal Context Teacher:
      tau^2 * KL(p_T1^tau || p_S^tau) + lambda_att * MSE(A_S, A_T1)
  T2 Local Action Discrimination Teacher:
      0.5 * sum_{l in {1,2}} MSE(phi_l(F_S^l), F_T2^l)
  T3 Human-Centered Teacher:
      MSE(psi(F_S^stage4), F_T3^ROI)

Implemented routing:
  teacher reliability r_i = exp(-CE(y, softmax(z_i/Tc)) / tau_r)
  d1 = variance of block-wise student fall probabilities
  d2 = entropy of student prediction
  d3 = KL(softmax(z_T3/Tc) || softmax(z_S/Tc))
  per-demand mini-batch z-score normalization
  q_i = log(r_i) + gamma * d_i_tilde
  alpha = softmax(q / tau_alpha), followed by stop-gradient

Overall objective:
  L_total = CE(y, p_S) + lambda_distill * sum_i alpha_i * L_Ti

Important compatibility notes:
1) The paper defines K temporal blocks but does not provide one fixed numerical K.
   Therefore real training REQUIRES --temporal-blocks K instead of silently inventing it.
2) The temporal-attention loss requires the final Stage-4 block of BOTH T1 and S to expose
   a SplitSABlock-style `t_attn` module with qkv/num_heads. If your config has SPLIT=False,
   this trainer stops with a clear error instead of silently changing the paper method.
3) The paper does not publish a complete dataset loader / project-specific model builder in
   the manuscript. Real mode therefore loads models through explicit module:function factories
   and consumes preprocessed full-frame/ROI clip tensors from a CSV manifest.
4) ROI clips should be produced with the paper's person-crop preprocessing. A missing ROI is
   NOT silently replaced by the full frame unless --allow-missing-roi is explicitly supplied.

Quick self-test (no dataset/checkpoints required):
  python train_rmtd_fd_distill.py --smoke-test

Real training manifest CSV:
  clip_path,roi_path,label
  clips/a.pt,rois/a.pt,1
  clips/b.pt,rois/b.pt,0

Each tensor file should contain one clip as [C,T,H,W] (preferred), [T,C,H,W], or
[T,H,W,C]. Supported file types: .pt/.pth/.npy/.npz. The trainer intentionally does
not impose an unreported normalization recipe; tensors should already use the same
preprocessing as the pretrained teachers/student.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility / utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_tensor(obj: Any, preferred_dim: Optional[int] = None) -> torch.Tensor:
    """Find the first useful tensor in nested model output."""
    if torch.is_tensor(obj):
        if preferred_dim is None or obj.dim() == preferred_dim:
            return obj
        raise TypeError(f"Tensor has dim={obj.dim()}, expected dim={preferred_dim}.")
    if isinstance(obj, Mapping):
        for key in ("logits", "pred", "prediction", "output", "x", "features", "feature"):
            if key in obj:
                try:
                    return _first_tensor(obj[key], preferred_dim)
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            try:
                return _first_tensor(value, preferred_dim)
            except (TypeError, ValueError):
                pass
    if isinstance(obj, (tuple, list)):
        for value in obj:
            try:
                return _first_tensor(value, preferred_dim)
            except (TypeError, ValueError):
                pass
    raise TypeError(f"Could not find a tensor in object of type {type(obj).__name__}.")


def extract_logits(output: Any, batch_size: int, num_classes: int = 2) -> torch.Tensor:
    """Extract [B, num_classes] logits from common model return structures."""
    candidates: List[torch.Tensor] = []

    def collect(x: Any) -> None:
        if torch.is_tensor(x):
            candidates.append(x)
        elif isinstance(x, Mapping):
            for key in ("logits", "pred", "prediction", "output"):
                if key in x:
                    collect(x[key])
            for value in x.values():
                collect(value)
        elif isinstance(x, (tuple, list)):
            for value in x:
                collect(value)

    collect(output)
    for tensor in candidates:
        if tensor.dim() == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_classes:
            return tensor
    shapes = [tuple(t.shape) for t in candidates]
    raise RuntimeError(
        f"Unable to locate logits with shape [B,{num_classes}] for batch B={batch_size}. "
        f"Tensor shapes returned by model: {shapes}"
    )


def find_stage_last_block(model: nn.Module, stage: int) -> nn.Module:
    attr = f"blocks{stage}"
    if not hasattr(model, attr):
        raise AttributeError(
            f"Model {type(model).__name__} has no `{attr}`. "
            "RMTD-FD feature hooks expect UniFormer-style blocks1/blocks2/blocks4 attributes."
        )
    blocks = getattr(model, attr)
    if isinstance(blocks, (nn.Sequential, nn.ModuleList, list, tuple)) and len(blocks) > 0:
        return blocks[-1]
    raise AttributeError(f"`{attr}` exists but is not a non-empty block sequence.")


def find_classifier_head(model: nn.Module) -> nn.Module:
    for name in ("head", "fc", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            return module
    raise AttributeError(
        f"Cannot find a student classification head on {type(model).__name__}; "
        "expected `.head`, `.fc`, or `.classifier`."
    )


def ensure_5d_feature(output: Any, source_name: str) -> torch.Tensor:
    try:
        x = _first_tensor(output, preferred_dim=5)
    except TypeError as exc:
        raise RuntimeError(
            f"{source_name} did not expose a 5-D [B,C,T,H,W] feature tensor."
        ) from exc
    return x


# -----------------------------
# Dataset
# -----------------------------

def _load_tensor_file(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch
            obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            tensor = obj
        elif isinstance(obj, Mapping):
            tensor = None
            for key in ("clip", "video", "frames", "tensor", "x"):
                if key in obj and torch.is_tensor(obj[key]):
                    tensor = obj[key]
                    break
            if tensor is None:
                raise ValueError(f"No tensor key found in {path}; tried clip/video/frames/tensor/x.")
        else:
            raise TypeError(f"Unsupported object stored in {path}: {type(obj).__name__}")
    elif suffix == ".npy":
        tensor = torch.from_numpy(np.load(path))
    elif suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise ValueError(f"Empty npz file: {path}")
        tensor = torch.from_numpy(data[data.files[0]])
    else:
        raise ValueError(f"Unsupported clip file extension: {path.suffix} ({path})")

    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 4:
        raise ValueError(f"Expected one 4-D clip in {path}, got shape {tuple(tensor.shape)}")

    # Convert common layouts to [C,T,H,W].
    if tensor.shape[0] in (1, 3):
        pass  # already [C,T,H,W]
    elif tensor.shape[1] in (1, 3):
        tensor = tensor.permute(1, 0, 2, 3)  # [T,C,H,W] -> [C,T,H,W]
    elif tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(3, 0, 1, 2)  # [T,H,W,C] -> [C,T,H,W]
    else:
        raise ValueError(
            f"Cannot infer channel dimension for {path}, shape={tuple(tensor.shape)}. "
            "Use [C,T,H,W], [T,C,H,W], or [T,H,W,C]."
        )

    tensor = tensor.contiguous()
    if not tensor.is_floating_point():
        tensor = tensor.float()
        if tensor.max().item() > 1.5:
            tensor = tensor / 255.0
    else:
        tensor = tensor.float()
    return tensor


class DistillManifestDataset(Dataset):
    """Manifest-backed preprocessed clip dataset."""

    def __init__(
        self,
        manifest: str,
        expected_frames: int = 16,
        expected_size: int = 224,
        allow_missing_roi: bool = False,
        strict_shape: bool = True,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.root = self.manifest.parent
        self.expected_frames = expected_frames
        self.expected_size = expected_size
        self.allow_missing_roi = allow_missing_roi
        self.strict_shape = strict_shape

        with self.manifest.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        required = {"clip_path", "label"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Manifest must contain columns {sorted(required)}; missing {sorted(missing)}")
        if "roi_path" not in rows[0] and not allow_missing_roi:
            raise ValueError(
                "Manifest has no `roi_path` column. The paper's T3 is an ROI teacher. "
                "Provide ROI clips, or explicitly opt into full-frame fallback with --allow-missing-roi."
            )
        self.rows = rows

    def _resolve(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    def _check_shape(self, clip: torch.Tensor, path: Path) -> None:
        if not self.strict_shape:
            return
        _, t, h, w = clip.shape
        if t != self.expected_frames or h != self.expected_size or w != self.expected_size:
            raise ValueError(
                f"{path} has [C,T,H,W]={tuple(clip.shape)}, but paper setting expects "
                f"T={self.expected_frames}, H=W={self.expected_size}. "
                "Preprocess clips before training or pass --no-strict-input-shape."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        clip_path = self._resolve(row["clip_path"])
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)
        clip = _load_tensor_file(clip_path)
        self._check_shape(clip, clip_path)

        roi_raw = (row.get("roi_path") or "").strip()
        if roi_raw:
            roi_path = self._resolve(roi_raw)
            if not roi_path.exists():
                if not self.allow_missing_roi:
                    raise FileNotFoundError(roi_path)
                roi = clip.clone()
            else:
                roi = _load_tensor_file(roi_path)
                self._check_shape(roi, roi_path)
        else:
            if not self.allow_missing_roi:
                raise ValueError(
                    f"Row {index + 2} has an empty roi_path. "
                    "Use the paper's ROI preprocessing, or pass --allow-missing-roi only for intended fallback cases."
                )
            roi = clip.clone()

        try:
            label = int(row["label"])
        except Exception as exc:
            raise ValueError(f"Invalid label at row {index + 2}: {row.get('label')!r}") from exc
        if label not in (0, 1):
            raise ValueError(f"Binary fall detection expects labels 0/1; got {label} at row {index + 2}")

        return {
            "clip": clip,
            "roi": roi,
            "label": torch.tensor(label, dtype=torch.long),
        }


# -----------------------------
# Model loading / forward adapter
# -----------------------------

def import_factory(spec: str):
    """Load `module:function` or `/path/file.py:function`."""
    if ":" not in spec:
        raise ValueError(f"Factory must be `module:function` or `/path/file.py:function`, got {spec!r}")
    module_spec, func_name = spec.rsplit(":", 1)
    if module_spec.endswith(".py") or os.path.sep in module_spec:
        path = Path(module_spec).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        module_name = f"rmtd_user_module_{abs(hash(str(path)))}"
        module_info = importlib.util.spec_from_file_location(module_name, path)
        if module_info is None or module_info.loader is None:
            raise ImportError(f"Cannot import module from {path}")
        module = importlib.util.module_from_spec(module_info)
        sys.modules[module_name] = module
        module_info.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    factory = getattr(module, func_name, None)
    if not callable(factory):
        raise AttributeError(f"{module_spec!r} has no callable {func_name!r}")
    return factory


def build_from_factory(spec: str, num_classes: int = 2) -> nn.Module:
    factory = import_factory(spec)
    attempts = [
        {"num_classes": num_classes},
        {"n_classes": num_classes},
        {"num_class": num_classes},
        {},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            model = factory(**kwargs)
            if not isinstance(model, nn.Module):
                raise TypeError(f"Factory returned {type(model).__name__}, expected torch.nn.Module")
            return model
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not call factory {spec!r}. Last error: {last_error}")


def unwrap_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in ("student", "model", "state_dict", "model_state_dict", "net", "network"):
            value = obj.get(key)
            if isinstance(value, Mapping) and any(torch.is_tensor(v) for v in value.values()):
                obj = value
                break
    if not isinstance(obj, Mapping) or not any(torch.is_tensor(v) for v in obj.values()):
        raise TypeError("Checkpoint does not contain a recognizable state_dict.")
    state = dict(obj)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def load_checkpoint_weights(model: nn.Module, path: str, strict: bool, name: str) -> None:
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    state = unwrap_state_dict(obj)
    result = model.load_state_dict(state, strict=strict)
    if not strict:
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            print(f"[{name}] non-strict checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing[:8]:
                print(f"  missing examples: {missing[:8]}")
            if unexpected[:8]:
                print(f"  unexpected examples: {unexpected[:8]}")


class ModelCaller:
    """Call models that expect either tensor clips or SlowFast-style [clip] lists."""

    def __init__(self, model: nn.Module, input_style: str, name: str, num_classes: int = 2):
        if input_style not in ("auto", "tensor", "list"):
            raise ValueError(input_style)
        self.model = model
        self.input_style = input_style
        self.name = name
        self.num_classes = num_classes
        self._resolved_style: Optional[str] = None if input_style == "auto" else input_style

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        batch_size = clip.shape[0]
        styles = [self._resolved_style] if self._resolved_style else ["list", "tensor"]
        errors: List[str] = []
        for style in styles:
            try:
                raw = self.model([clip]) if style == "list" else self.model(clip)
                logits = extract_logits(raw, batch_size=batch_size, num_classes=self.num_classes)
                if self._resolved_style is None:
                    self._resolved_style = style
                    print(f"[{self.name}] auto-detected input style: {style}")
                return logits
            except Exception as exc:
                errors.append(f"{style}: {type(exc).__name__}: {exc}")
                if self._resolved_style is not None:
                    break
        raise RuntimeError(f"{self.name} forward failed. Attempts -> " + " | ".join(errors))


# -----------------------------
# Feature / attention hooks
# -----------------------------

class FeatureCapture:
    def __init__(self, module: nn.Module, name: str):
        self.name = name
        self.value: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        self.value = ensure_5d_feature(output, self.name)

    def get(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError(f"No feature captured for {self.name}; hook target may not have executed.")
        return self.value

    def close(self) -> None:
        self.handle.remove()


class TemporalAttentionCapture:
    """
    Reconstructs attention immediately after softmax and before attention dropout from
    a SplitSABlock-style temporal attention module. The expected temporal-attention
    input is [B*spatial_tokens, T, C].
    """

    def __init__(self, block: nn.Module, name: str):
        self.name = name
        self.current_batch_size: Optional[int] = None
        self.value: Optional[torch.Tensor] = None
        t_attn = getattr(block, "t_attn", None)
        if not isinstance(t_attn, nn.Module):
            raise RuntimeError(
                f"{name}: final Stage-4 block is {type(block).__name__} and has no `.t_attn`. "
                "The paper explicitly distills temporal attention from the last Stage-4 SplitSABlock. "
                "Use a model/config with the temporal SplitSABlock enabled (commonly SPLIT=True)."
            )
        for attr in ("qkv", "num_heads"):
            if not hasattr(t_attn, attr):
                raise RuntimeError(
                    f"{name}: `.t_attn` lacks `{attr}`, so pre-dropout temporal attention cannot be "
                    "reconstructed without changing the paper's distillation signal."
                )
        self.t_attn = t_attn
        self.handle = t_attn.register_forward_pre_hook(self._pre_hook)

    def set_batch_size(self, batch_size: int) -> None:
        self.current_batch_size = int(batch_size)
        self.value = None

    def _pre_hook(self, module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if self.current_batch_size is None:
            raise RuntimeError(f"{self.name}: batch size was not set before temporal attention forward.")
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError(f"{self.name}: unexpected t_attn input structure.")
        x = inputs[0]
        if x.dim() != 3:
            raise RuntimeError(
                f"{self.name}: expected t_attn input [B*HW,T,C], got {tuple(x.shape)}."
            )
        n, t, c = x.shape
        heads = int(module.num_heads)
        if c % heads != 0:
            raise RuntimeError(f"{self.name}: channels C={c} not divisible by num_heads={heads}.")
        qkv = module.qkv(x)
        if qkv.shape[-1] != 3 * c:
            raise RuntimeError(
                f"{self.name}: qkv output last dim={qkv.shape[-1]}, expected {3*c}."
            )
        head_dim = c // heads
        qkv = qkv.reshape(n, t, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = float(getattr(module, "scale", head_dim ** -0.5))
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        b = self.current_batch_size
        if n % b != 0:
            raise RuntimeError(
                f"{self.name}: t_attn leading dimension N={n} is not divisible by batch B={b}; "
                "cannot average over spatial tokens as specified."
            )
        spatial = n // b
        # [B, spatial, heads, T, T] -> average over spatial and heads.
        self.value = attn.reshape(b, spatial, heads, t, t).mean(dim=(1, 2))

    def get(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError(f"No temporal attention captured for {self.name}.")
        return self.value

    def close(self) -> None:
        self.handle.remove()


# -----------------------------
# Paper losses / dynamic routing
# -----------------------------

def per_sample_mse(a: torch.Tensor, b: torch.Tensor, name: str) -> torch.Tensor:
    if a.shape != b.shape:
        raise RuntimeError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a - b).pow(2).flatten(1).mean(dim=1)


def kd_kl_teacher_to_student(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)
    return (temperature ** 2) * kl


def categorical_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * p.clamp_min(eps).log()).sum(dim=1)


def categorical_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_p_teacher = p_teacher.clamp_min(1e-12).log()
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    return (p_teacher * (log_p_teacher - log_p_student)).sum(dim=1)


def teacher_reliability(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    calibration_temperature: float,
    reliability_temperature: float,
) -> torch.Tensor:
    p = F.softmax(teacher_logits / calibration_temperature, dim=1)
    p_y = p.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
    e = -p_y.log()
    return torch.exp(-e / reliability_temperature)


def batch_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)


def temporal_role_demand(
    stage4_feature: torch.Tensor,
    classifier_head: nn.Module,
    temporal_blocks: int,
    fall_class_index: int,
) -> torch.Tensor:
    """
    Single-forward block-wise prediction fluctuation.

    The paper defines K temporal blocks but does not report a fixed K. This implementation
    uses the captured pre-pooling Stage-4 student feature so the already-trained student
    classification head can be reused without introducing a new unreported classifier.
    """
    if stage4_feature.dim() != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(stage4_feature.shape)}")
    b, c, t, h, w = stage4_feature.shape
    if temporal_blocks < 2:
        raise ValueError("--temporal-blocks must be >= 2 for a variance-based demand signal.")
    if temporal_blocks > t:
        raise ValueError(
            f"--temporal-blocks={temporal_blocks} exceeds Stage-4 temporal tokens T={t}. "
            "Choose K <= the actual Stage-4 temporal token count."
        )
    chunks = torch.tensor_split(stage4_feature, temporal_blocks, dim=2)
    probs: List[torch.Tensor] = []
    for chunk in chunks:
        pooled = chunk.mean(dim=(2, 3, 4))  # [B,C]
        logits = classifier_head(pooled)
        logits = extract_logits(logits, batch_size=b, num_classes=2)
        probs.append(F.softmax(logits, dim=1)[:, fall_class_index])
    block_probs = torch.stack(probs, dim=1)  # [B,K]
    return block_probs.var(dim=1, unbiased=False)


class FeatureProjectors(nn.Module):
    """Paper's learnable 1x1x1 alignment convolutions phi_1, phi_2, psi."""

    def __init__(self, s1: int, t1: int, s2: int, t2: int, s4: int, t3: int):
        super().__init__()
        self.phi1 = nn.Conv3d(s1, t1, kernel_size=1, bias=False)
        self.phi2 = nn.Conv3d(s2, t2, kernel_size=1, bias=False)
        self.psi = nn.Conv3d(s4, t3, kernel_size=1, bias=False)

    @staticmethod
    def from_features(
        s1: torch.Tensor,
        t2_1: torch.Tensor,
        s2: torch.Tensor,
        t2_2: torch.Tensor,
        s4: torch.Tensor,
        t3_4: torch.Tensor,
    ) -> "FeatureProjectors":
        return FeatureProjectors(
            s1=s1.shape[1], t1=t2_1.shape[1],
            s2=s2.shape[1], t2=t2_2.shape[1],
            s4=s4.shape[1], t3=t3_4.shape[1],
        )


def check_grid_match(projected: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if projected.shape[0] != target.shape[0] or projected.shape[2:] != target.shape[2:]:
        raise RuntimeError(
            f"{name}: 1x1x1 projection can align channels only, but grids differ: "
            f"projected={tuple(projected.shape)}, target={tuple(target.shape)}. "
            "The paper specifies a 1x1x1 alignment convolution, so this trainer does not "
            "silently interpolate features. Check model/input configuration."
        )


@dataclass
class PaperHyperParams:
    tau: float = 4.0
    lambda_distill: float = 0.5
    lambda_att: float = 1.0
    tau_r: float = 1.0
    gamma: float = 0.5
    tau_alpha: float = 1.0
    tc: float = 2.0
    fall_class_index: int = 1
    temporal_blocks: int = 0
    eps: float = 1e-6


class RMTDFDTrainerCore(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher_t1: nn.Module,
        teacher_t2: nn.Module,
        teacher_t3: nn.Module,
        hp: PaperHyperParams,
        student_input_style: str = "auto",
        t1_input_style: str = "auto",
        t2_input_style: str = "auto",
        t3_input_style: str = "auto",
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher_t1 = teacher_t1
        self.teacher_t2 = teacher_t2
        self.teacher_t3 = teacher_t3
        self.hp = hp
        self.projectors: Optional[FeatureProjectors] = None

        # Teachers are frozen exactly as in the distillation stage described in the paper.
        for teacher in (teacher_t1, teacher_t2, teacher_t3):
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        self.s_call = ModelCaller(student, student_input_style, "Student")
        self.t1_call = ModelCaller(teacher_t1, t1_input_style, "T1-Temporal")
        self.t2_call = ModelCaller(teacher_t2, t2_input_style, "T2-Local")
        self.t3_call = ModelCaller(teacher_t3, t3_input_style, "T3-ROI")

        # UniFormer stage hooks required by Eqs. (4)-(7).
        self.s_f1 = FeatureCapture(find_stage_last_block(student, 1), "Student Stage1")
        self.s_f2 = FeatureCapture(find_stage_last_block(student, 2), "Student Stage2")
        self.s_f4 = FeatureCapture(find_stage_last_block(student, 4), "Student Stage4")
        self.t2_f1 = FeatureCapture(find_stage_last_block(teacher_t2, 1), "T2 Stage1")
        self.t2_f2 = FeatureCapture(find_stage_last_block(teacher_t2, 2), "T2 Stage2")
        self.t3_f4 = FeatureCapture(find_stage_last_block(teacher_t3, 4), "T3 Stage4")

        # Paper: last SplitSABlock of Stage4 in T1 and S.
        self.s_att = TemporalAttentionCapture(find_stage_last_block(student, 4), "Student Stage4 attention")
        self.t1_att = TemporalAttentionCapture(find_stage_last_block(teacher_t1, 4), "T1 Stage4 attention")
        self.student_head = find_classifier_head(student)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep all teachers fixed/eval even when trainer.train() is called.
        self.teacher_t1.eval()
        self.teacher_t2.eval()
        self.teacher_t3.eval()
        return self

    def _forward_all(self, clip: torch.Tensor, roi: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = clip.shape[0]
        self.s_att.set_batch_size(b)
        self.t1_att.set_batch_size(b)

        s_logits = self.s_call(clip)
        s1, s2, s4 = self.s_f1.get(), self.s_f2.get(), self.s_f4.get()
        s_att = self.s_att.get()

        with torch.no_grad():
            t1_logits = self.t1_call(clip)
            t1_att = self.t1_att.get()
            t2_logits = self.t2_call(clip)
            t2_1, t2_2 = self.t2_f1.get(), self.t2_f2.get()
            t3_logits = self.t3_call(roi)
            t3_4 = self.t3_f4.get()

        return {
            "s_logits": s_logits,
            "s1": s1,
            "s2": s2,
            "s4": s4,
            "s_att": s_att,
            "t1_logits": t1_logits,
            "t1_att": t1_att,
            "t2_logits": t2_logits,
            "t2_1": t2_1,
            "t2_2": t2_2,
            "t3_logits": t3_logits,
            "t3_4": t3_4,
        }

    @torch.no_grad()
    def materialize_projectors(self, batch: Mapping[str, torch.Tensor], device: torch.device) -> None:
        clip = batch["clip"].to(device, non_blocking=True)
        roi = batch["roi"].to(device, non_blocking=True)
        was_training = self.student.training
        self.student.eval()
        out = self._forward_all(clip, roi)
        self.projectors = FeatureProjectors.from_features(
            out["s1"], out["t2_1"], out["s2"], out["t2_2"], out["s4"], out["t3_4"]
        ).to(device)
        # Validate the paper's channel-only alignment assumption immediately.
        p1 = self.projectors.phi1(out["s1"])
        p2 = self.projectors.phi2(out["s2"])
        p3 = self.projectors.psi(out["s4"])
        check_grid_match(p1, out["t2_1"], "T2 Stage1")
        check_grid_match(p2, out["t2_2"], "T2 Stage2")
        check_grid_match(p3, out["t3_4"], "T3 Stage4")
        if was_training:
            self.student.train()
        print(
            "[Projectors] materialized: "
            f"phi1 {out['s1'].shape[1]}->{out['t2_1'].shape[1]}, "
            f"phi2 {out['s2'].shape[1]}->{out['t2_2'].shape[1]}, "
            f"psi {out['s4'].shape[1]}->{out['t3_4'].shape[1]}"
        )

    def compute_loss(self, batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        if self.projectors is None:
            raise RuntimeError("Feature projectors have not been materialized.")
        clip = batch["clip"].to(device, non_blocking=True)
        roi = batch["roi"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        hp = self.hp

        out = self._forward_all(clip, roi)

        # T1: soft-label + temporal-attention supervision.
        l_t1_kd = kd_kl_teacher_to_student(out["t1_logits"], out["s_logits"], hp.tau)
        l_t1_att = per_sample_mse(out["s_att"], out["t1_att"], "T1 temporal attention")
        l_t1 = l_t1_kd + hp.lambda_att * l_t1_att

        # T2: Stage1/2 local feature alignment through learnable 1x1x1 convolutions.
        s1p = self.projectors.phi1(out["s1"])
        s2p = self.projectors.phi2(out["s2"])
        check_grid_match(s1p, out["t2_1"], "T2 Stage1")
        check_grid_match(s2p, out["t2_2"], "T2 Stage2")
        l_t2 = 0.5 * (
            per_sample_mse(s1p, out["t2_1"], "T2 Stage1")
            + per_sample_mse(s2p, out["t2_2"], "T2 Stage2")
        )

        # T3: Stage4 full-frame student vs ROI-teacher human-centered features.
        s4p = self.projectors.psi(out["s4"])
        check_grid_match(s4p, out["t3_4"], "T3 Stage4")
        l_t3 = per_sample_mse(s4p, out["t3_4"], "T3 Stage4")

        # Reliability-demand routing is used only as a detached routing coefficient.
        with torch.no_grad():
            r1 = teacher_reliability(out["t1_logits"], labels, hp.tc, hp.tau_r)
            r2 = teacher_reliability(out["t2_logits"], labels, hp.tc, hp.tau_r)
            r3 = teacher_reliability(out["t3_logits"], labels, hp.tc, hp.tau_r)

            d1 = temporal_role_demand(
                out["s4"].detach(), self.student_head, hp.temporal_blocks, hp.fall_class_index
            )
            d2 = categorical_entropy_from_logits(out["s_logits"].detach())
            d3 = categorical_kl_from_logits(out["t3_logits"], out["s_logits"].detach(), hp.tc)

            d = torch.stack((batch_zscore(d1, hp.eps), batch_zscore(d2, hp.eps), batch_zscore(d3, hp.eps)), dim=1)
            r = torch.stack((r1, r2, r3), dim=1)
            q = r.clamp_min(1e-12).log() + hp.gamma * d
            alpha = F.softmax(q / hp.tau_alpha, dim=1).detach()

        per_teacher = torch.stack((l_t1, l_t2, l_t3), dim=1)
        l_mtd_per_sample = (alpha * per_teacher).sum(dim=1)
        l_mtd = l_mtd_per_sample.mean()
        l_ce = F.cross_entropy(out["s_logits"], labels)
        l_total = l_ce + hp.lambda_distill * l_mtd

        return {
            "loss": l_total,
            "ce": l_ce.detach(),
            "mtd": l_mtd.detach(),
            "t1": l_t1.mean().detach(),
            "t1_kd": l_t1_kd.mean().detach(),
            "t1_att": l_t1_att.mean().detach(),
            "t2": l_t2.mean().detach(),
            "t3": l_t3.mean().detach(),
            "alpha1": alpha[:, 0].mean(),
            "alpha2": alpha[:, 1].mean(),
            "alpha3": alpha[:, 2].mean(),
            "alpha": alpha,
            "d1": d1.mean(),
            "d2": d2.mean(),
            "d3": d3.mean(),
            "logits": out["s_logits"],
            "labels": labels,
        }


# -----------------------------
# Metrics / training loop
# -----------------------------

class BinaryMeter:
    def __init__(self, fall_class_index: int = 1):
        self.pos = fall_class_index
        self.tp = self.fp = self.tn = self.fn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        y_pos = labels.eq(self.pos)
        p_pos = pred.eq(self.pos)
        self.tp += int((p_pos & y_pos).sum().item())
        self.fp += int((p_pos & ~y_pos).sum().item())
        self.tn += int((~p_pos & ~y_pos).sum().item())
        self.fn += int((~p_pos & y_pos).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-12
        total = self.tp + self.fp + self.tn + self.fn
        acc = (self.tp + self.tn) / max(total, 1)
        precision = self.tp / max(self.tp + self.fp, eps)
        recall = self.tp / max(self.tp + self.fn, eps)
        f1 = 2 * precision * recall / max(precision + recall, eps)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class MeanTracker:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Mapping[str, torch.Tensor], n: int) -> None:
        keys = ("loss", "ce", "mtd", "t1", "t1_kd", "t1_att", "t2", "t3", "alpha1", "alpha2", "alpha3", "d1", "d2", "d3")
        for key in keys:
            if key in values:
                value = values[key]
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: float("nan") for k in self.sums}
        return {k: v / self.count for k, v in self.sums.items()}


def train_one_epoch(
    core: RMTDFDTrainerCore,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    grad_clip: float,
    log_interval: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    core.train()
    meter = MeanTracker()
    metric = BinaryMeter(core.hp.fall_class_index)
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = core.compute_loss(batch, device)
            loss = out["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in core.parameters() if p.requires_grad], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in core.parameters() if p.requires_grad], grad_clip
                )
            optimizer.step()

        n = int(out["labels"].shape[0])
        meter.update(out, n=n)
        metric.update(out["logits"].detach(), out["labels"])

        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == len(loader)):
            avg = meter.compute()
            elapsed = time.time() - start
            print(
                f"  step {step:4d}/{len(loader):4d} "
                f"loss={avg.get('loss', float('nan')):.4f} "
                f"CE={avg.get('ce', float('nan')):.4f} "
                f"MTD={avg.get('mtd', float('nan')):.4f} "
                f"alpha=({avg.get('alpha1',0):.3f},{avg.get('alpha2',0):.3f},{avg.get('alpha3',0):.3f}) "
                f"time={elapsed:.1f}s"
            )
    return meter.compute(), metric.compute()


@torch.no_grad()
def validate_student(
    student: nn.Module,
    caller: ModelCaller,
    loader: DataLoader,
    device: torch.device,
    fall_class_index: int,
) -> Dict[str, float]:
    student.eval()
    metric = BinaryMeter(fall_class_index)
    ce_sum = 0.0
    count = 0
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = caller(clip)
        ce_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        count += labels.numel()
        metric.update(logits, labels)
    result = metric.compute()
    result["ce"] = ce_sum / max(count, 1)
    return result


def save_student_checkpoint(
    path: Path,
    core: RMTDFDTrainerCore,
    epoch: int,
    val_metrics: Optional[Mapping[str, float]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "student": core.student.state_dict(),
        "val_metrics": dict(val_metrics or {}),
        "paper_hparams": asdict(core.hp),
        "args": vars(args),
    }
    torch.save(payload, path)


def save_training_checkpoint(
    path: Path,
    core: RMTDFDTrainerCore,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_f1: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "student": core.student.state_dict(),
        "projectors": core.projectors.state_dict() if core.projectors is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_f1": best_f1,
        "paper_hparams": asdict(core.hp),
        "args": vars(args),
    }
    torch.save(payload, path)


def load_resume(
    path: str,
    core: RMTDFDTrainerCore,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Tuple[int, float]:
    p = Path(path).expanduser().resolve()
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")
    core.student.load_state_dict(ckpt["student"], strict=True)
    if core.projectors is not None and ckpt.get("projectors") is not None:
        core.projectors.load_state_dict(ckpt["projectors"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_f1", -1.0))


# -----------------------------
# Tiny smoke-test models
# -----------------------------

class TinyTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, c = x.shape
        qkv = self.qkv(x).reshape(n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(n, t, c)
        return self.proj(y)


class TinyConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class TinySplitSABlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        heads = 2 if dim % 2 == 0 else 1
        self.t_attn = TinyTemporalAttention(dim, num_heads=heads)
        self.ffn = nn.Conv3d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        # Temporal attention independently at each spatial token.
        xt = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        xt = self.t_attn(xt)
        y = xt.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()
        return x + self.ffn(y)


class TinyUniformer(nn.Module):
    def __init__(self, channels: Sequence[int], num_classes: int = 2):
        super().__init__()
        c1, c2, c4 = channels
        self.blocks1 = nn.ModuleList([TinyConvBlock(3, c1)])
        self.blocks2 = nn.ModuleList([TinyConvBlock(c1, c2)])
        self.bridge = TinyConvBlock(c2, c4)
        self.blocks4 = nn.ModuleList([TinySplitSABlock(c4)])
        self.head = nn.Linear(c4, num_classes)

    def forward(self, x: Any) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.blocks1[0](x)
        x = self.blocks2[0](x)
        x = self.bridge(x)
        x = self.blocks4[0](x)
        return self.head(x.mean(dim=(2, 3, 4)))


def run_smoke_test() -> None:
    print("[Smoke test] Building tiny paper-shaped student/teachers on CPU...")
    device = torch.device("cpu")
    set_seed(7)
    student = TinyUniformer((4, 6, 8))
    t1 = TinyUniformer((6, 8, 12))
    t2 = TinyUniformer((5, 9, 10))
    t3 = TinyUniformer((7, 11, 14))
    hp = PaperHyperParams(temporal_blocks=2)
    core = RMTDFDTrainerCore(student, t1, t2, t3, hp).to(device)

    batch = {
        "clip": torch.randn(4, 3, 4, 8, 8),
        "roi": torch.randn(4, 3, 4, 8, 8),
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    core.materialize_projectors(batch, device)
    optimizer = torch.optim.AdamW(
        [p for p in core.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05
    )
    core.train()
    out = core.compute_loss(batch, device)
    assert torch.isfinite(out["loss"]), "loss is not finite"
    assert torch.allclose(out["alpha"].sum(dim=1), torch.ones(4), atol=1e-6), "alpha rows do not sum to 1"
    assert not out["alpha"].requires_grad, "alpha must be stop-gradient"
    optimizer.zero_grad(set_to_none=True)
    out["loss"].backward()
    # Confirm student/projector gradients exist and teacher gradients do not.
    student_grad = any(p.grad is not None for p in core.student.parameters() if p.requires_grad)
    proj_grad = any(p.grad is not None for p in core.projectors.parameters() if p.requires_grad)
    teacher_grad = any(
        p.grad is not None
        for teacher in (core.teacher_t1, core.teacher_t2, core.teacher_t3)
        for p in teacher.parameters()
    )
    assert student_grad and proj_grad, "student/projector gradients missing"
    assert not teacher_grad, "teacher gradients should be frozen"
    optimizer.step()
    print(
        "[Smoke test] PASS | "
        f"loss={out['loss'].item():.4f} | "
        f"alpha_mean=({out['alpha1'].item():.3f}, {out['alpha2'].item():.3f}, {out['alpha3'].item():.3f})"
    )


# -----------------------------
# CLI
# -----------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RMTD-FD paper-faithful multi-teacher distillation trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoke-test", action="store_true", help="Run a self-contained CPU test and exit.")

    # Data
    p.add_argument("--train-manifest", type=str, default=None)
    p.add_argument("--val-manifest", type=str, default=None)
    p.add_argument("--allow-missing-roi", action="store_true")
    p.add_argument("--no-strict-input-shape", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")

    # Model factories / checkpoints
    p.add_argument("--student-factory", type=str, default=None)
    p.add_argument("--t1-factory", type=str, default=None, help="Temporal teacher (paper: UniFormer-B).")
    p.add_argument("--t2-factory", type=str, default=None, help="Local teacher (paper: UniFormer-S).")
    p.add_argument("--t3-factory", type=str, default=None, help="ROI teacher (paper: ROI-UniFormer-B).")
    p.add_argument("--student-ckpt", type=str, default=None, help="Optional student initialization.")
    p.add_argument("--t1-ckpt", type=str, default=None)
    p.add_argument("--t2-ckpt", type=str, default=None)
    p.add_argument("--t3-ckpt", type=str, default=None)
    p.add_argument("--strict-checkpoint", action="store_true")
    for name in ("student", "t1", "t2", "t3"):
        p.add_argument(f"--{name}-input-style", choices=("auto", "list", "tensor"), default="auto")

    # Paper implementation details / hyperparameters
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8, help="Paper distillation batch size.")
    p.add_argument("--epochs", type=int, default=100, help="Paper distillation epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Paper distillation initial learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=4.0, help="Distillation temperature.")
    p.add_argument("--lambda-distill", type=float, default=0.5, help="Overall distillation balance coefficient lambda.")
    p.add_argument("--lambda-att", type=float, default=1.0)
    p.add_argument("--tau-r", type=float, default=1.0, help="Reliability temperature.")
    p.add_argument("--gamma", type=float, default=0.5, help="Role-demand contribution.")
    p.add_argument("--tau-alpha", type=float, default=1.0, help="Teacher weight-allocation temperature.")
    p.add_argument("--tc", type=float, default=2.0, help="Teacher calibration temperature.")
    p.add_argument("--temporal-blocks", type=int, default=None, help="K in the paper's temporal role demand. Required in real mode because the paper does not fix K numerically.")
    p.add_argument("--fall-class-index", type=int, choices=(0, 1), default=1)

    # Runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Use CUDA AMP when running on CUDA.")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="./runs/rmtd_fd_distill")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default=None)
    return p


def validate_real_args(args: argparse.Namespace) -> None:
    required = {
        "--train-manifest": args.train_manifest,
        "--student-factory": args.student_factory,
        "--t1-factory": args.t1_factory,
        "--t2-factory": args.t2_factory,
        "--t3-factory": args.t3_factory,
        "--t1-ckpt": args.t1_ckpt,
        "--t2-ckpt": args.t2_ckpt,
        "--t3-ckpt": args.t3_ckpt,
        "--temporal-blocks": args.temporal_blocks,
    }
    missing = [k for k, v in required.items() if v is None or v == ""]
    if missing:
        raise SystemExit(
            "Real training is missing required arguments: " + ", ".join(missing) + "\n"
            "Use --smoke-test first if you only want to verify that this training file itself runs."
        )
    if args.temporal_blocks is not None and args.temporal_blocks < 2:
        raise SystemExit("--temporal-blocks must be >= 2")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
        return

    validate_real_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    print("=== RMTD-FD Distillation ===")
    print(f"Device: {device}")
    print("Paper defaults: 16 frames, 224x224, AdamW, lr=1e-4, wd=0.05, batch=8, epochs=100")
    print(
        f"Routing/loss: tau={args.tau}, lambda={args.lambda_distill}, lambda_att={args.lambda_att}, "
        f"tau_r={args.tau_r}, gamma={args.gamma}, tau_alpha={args.tau_alpha}, Tc={args.tc}, K={args.temporal_blocks}"
    )
    print(
        "Implementation note: d1 reuses the student's classifier on pre-pooling Stage-4 temporal blocks; "
        "the manuscript defines K/block-wise fluctuation but does not fix K numerically."
    )

    train_ds = DistillManifestDataset(
        args.train_manifest,
        expected_frames=args.num_frames,
        expected_size=args.image_size,
        allow_missing_roi=args.allow_missing_roi,
        strict_shape=not args.no_strict_input_shape,
    )
    val_ds = None
    if args.val_manifest:
        val_ds = DistillManifestDataset(
            args.val_manifest,
            expected_frames=args.num_frames,
            expected_size=args.image_size,
            allow_missing_roi=args.allow_missing_roi,
            strict_shape=not args.no_strict_input_shape,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

    print("Building models from factories...")
    student = build_from_factory(args.student_factory).to(device)
    t1 = build_from_factory(args.t1_factory).to(device)
    t2 = build_from_factory(args.t2_factory).to(device)
    t3 = build_from_factory(args.t3_factory).to(device)

    if args.student_ckpt:
        load_checkpoint_weights(student, args.student_ckpt, args.strict_checkpoint, "Student")
    load_checkpoint_weights(t1, args.t1_ckpt, args.strict_checkpoint, "T1")
    load_checkpoint_weights(t2, args.t2_ckpt, args.strict_checkpoint, "T2")
    load_checkpoint_weights(t3, args.t3_ckpt, args.strict_checkpoint, "T3")

    hp = PaperHyperParams(
        tau=args.tau,
        lambda_distill=args.lambda_distill,
        lambda_att=args.lambda_att,
        tau_r=args.tau_r,
        gamma=args.gamma,
        tau_alpha=args.tau_alpha,
        tc=args.tc,
        fall_class_index=args.fall_class_index,
        temporal_blocks=args.temporal_blocks,
    )
    core = RMTDFDTrainerCore(
        student, t1, t2, t3, hp,
        student_input_style=args.student_input_style,
        t1_input_style=args.t1_input_style,
        t2_input_style=args.t2_input_style,
        t3_input_style=args.t3_input_style,
    ).to(device)

    # Materialize the paper's learnable 1x1x1 projectors before constructing AdamW.
    try:
        first_batch = next(iter(train_loader))
    except StopIteration:
        raise SystemExit("Training dataset is empty.")
    core.materialize_projectors(first_batch, device)

    trainable = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    start_epoch = 1
    best_f1 = -1.0
    if args.resume:
        start_epoch, best_f1 = load_resume(args.resume, core, optimizer, scaler)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, best_f1={best_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_one_epoch(
            core, train_loader, optimizer, device, scaler, use_amp,
            grad_clip=args.grad_clip, log_interval=args.log_interval,
        )
        print(
            "Train | "
            f"loss={train_loss.get('loss', float('nan')):.4f} "
            f"Acc={train_metrics['accuracy']*100:.2f}% "
            f"P={train_metrics['precision']*100:.2f}% "
            f"R={train_metrics['recall']*100:.2f}% "
            f"F1={train_metrics['f1']*100:.2f}%"
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = validate_student(core.student, core.s_call, val_loader, device, args.fall_class_index)
            print(
                "Val   | "
                f"CE={val_metrics['ce']:.4f} "
                f"Acc={val_metrics['accuracy']*100:.2f}% "
                f"P={val_metrics['precision']*100:.2f}% "
                f"R={val_metrics['recall']*100:.2f}% "
                f"F1={val_metrics['f1']*100:.2f}%"
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_student_checkpoint(output_dir / "best_student.pt", core, epoch, val_metrics, args)
                print(f"Saved new best_student.pt (F1={best_f1*100:.2f}%)")
        else:
            # Without validation, keep the latest student as the best-available artifact.
            save_student_checkpoint(output_dir / "best_student.pt", core, epoch, None, args)

        save_training_checkpoint(
            output_dir / "last_training.pt", core, optimizer, scaler, epoch, best_f1, args
        )

    print("\nTraining complete.")
    print(f"Outputs: {output_dir}")
    print("Inference should retain only the student weights; teachers/routing/projectors are training-only.")


if __name__ == "__main__":
    main()
