#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMTD-FD inference + online fall alarm
=====================================

This file implements the inference protocol described in the RMTD-FD paper.

Paper-faithful inference
------------------------
- Only the lightweight student model (UniFormer-XS) is retained at inference.
- No teacher network is loaded.
- No ROI teacher branch is used.
- No dynamic teacher weighting / routing is computed.
- Continuous RGB input is converted to sliding-window clips.
- Paper defaults:
      clip length L = 16 frames
      stride s      = 8 frames
      input size    = 224 x 224
      moving average M = 3 windows
      persistence C    = 2 consecutive windows
      alarm threshold theta = 0.65
- For each sliding window:
      p_t = fall probability of the student
      p_bar_t = causal moving average of the most recent M probabilities
- A fall alarm is triggered when p_bar_t >= theta for C consecutive windows.
- Consecutive alarm windows are merged into one alarm event.
- The first window at which the persistence condition is satisfied is recorded
  as the alarm trigger time.

Important implementation note
-----------------------------
The paper does not specify the exact image normalization constants used by the
local training code. To avoid silently inventing them, this script defaults to:
    BGR frame -> RGB -> resize 224x224 -> float32 / 255
which is consistent with the preprocessing/training files generated alongside
this project.

If your actual UniFormer implementation used mean/std normalization, pass:
    --mean m1 m2 m3 --std s1 s2 s3
using the SAME values used during training.

Supported input
---------------
1) Video file:
    --source /path/to/video.mp4

2) Webcam / camera:
    --source 0
    --source 1

Output
------
output_dir/
    window_predictions.csv
    alarm_events.csv
    inference_summary.json
    annotated_output.mp4      (only when --save-video is used)

Quick self-test
---------------
    python inference_online_alarm_rmtd_fd.py --smoke-test

Example
-------
    python inference_online_alarm_rmtd_fd.py \
        --source sample.mp4 \
        --model-factory models.uniformer:uniformer_xs \
        --checkpoint runs/distill/best_student.pt \
        --output-dir runs/online_demo \
        --display \
        --save-video \
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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Model factory + checkpoint loader
# ---------------------------------------------------------------------

def import_factory(spec: str):
    """
    Load:
        package.module:function
    or:
        /path/to/file.py:function
    """
    if ":" not in spec:
        raise ValueError(
            f"Factory must be `module:function` or `file.py:function`, got {spec!r}"
        )

    module_spec, function_name = spec.rsplit(":", 1)

    if module_spec.endswith(".py") or os.path.sep in module_spec:
        module_path = Path(module_spec).expanduser().resolve()
        if not module_path.exists():
            raise FileNotFoundError(module_path)

        module_name = f"rmtd_infer_factory_{abs(hash(str(module_path)))}"
        module_info = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if module_info is None or module_info.loader is None:
            raise ImportError(f"Cannot import model file: {module_path}")

        module = importlib.util.module_from_spec(module_info)
        sys.modules[module_name] = module
        module_info.loader.exec_module(module)

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
                    f"Factory returned {type(model).__name__}; expected nn.Module."
                )
            return model
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not construct the student with {factory_spec!r}. "
        f"Last error: {last_error}"
    )


def unwrap_student_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """
    Supports the checkpoint formats generated by train_rmtd_fd_distill.py:
        {"student": state_dict, ...}
    and common PyTorch formats.
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
        raise TypeError(
            "Checkpoint does not contain a recognizable student state_dict."
        )

    state = {
        k: v for k, v in obj.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }

    if not state:
        raise TypeError("Student state_dict is empty or unrecognized.")

    if all(k.startswith("module.") for k in state):
        state = {
            k[len("module."):]: v
            for k, v in state.items()
        }

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
        raw = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        raw = torch.load(
            path,
            map_location="cpu",
        )

    state = unwrap_student_state_dict(raw)
    result = model.load_state_dict(
        state,
        strict=strict,
    )

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
# Generic student forward adapter
# ---------------------------------------------------------------------

def _collect_tensors(obj: Any, out: List[torch.Tensor]) -> None:
    if torch.is_tensor(obj):
        out.append(obj)

    elif isinstance(obj, Mapping):
        # Prefer conventional prediction keys first.
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

    for tensor in tensors:
        if (
            tensor.dim() == 2
            and tensor.shape[0] == batch_size
            and tensor.shape[1] == num_classes
        ):
            return tensor

    shapes = [tuple(t.shape) for t in tensors]

    raise RuntimeError(
        f"Could not find logits [B,{num_classes}] for B={batch_size}. "
        f"Returned tensor shapes: {shapes}"
    )


class ModelCaller:
    """
    Supports common UniFormer-style calls:
        model(clip)
        model([clip])

    `auto` tries list first and then tensor, then remembers the working style.
    """

    def __init__(
        self,
        model: nn.Module,
        input_style: str = "auto",
        num_classes: int = 2,
    ) -> None:
        if input_style not in ("auto", "tensor", "list"):
            raise ValueError(
                "input_style must be one of: auto, tensor, list"
            )

        self.model = model
        self.num_classes = num_classes
        self.resolved: Optional[str] = (
            None if input_style == "auto" else input_style
        )

    def __call__(
        self,
        clip: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = clip.shape[0]

        styles = (
            [self.resolved]
            if self.resolved is not None
            else ["list", "tensor"]
        )

        errors: List[str] = []

        for style in styles:
            try:
                raw = (
                    self.model([clip])
                    if style == "list"
                    else self.model(clip)
                )

                logits = extract_logits(
                    raw,
                    batch_size=batch_size,
                    num_classes=self.num_classes,
                )

                if self.resolved is None:
                    self.resolved = style
                    print(
                        f"[model] auto-detected input style: {style}"
                    )

                return logits

            except Exception as exc:
                errors.append(
                    f"{style}: {type(exc).__name__}: {exc}"
                )

                if self.resolved is not None:
                    break

        raise RuntimeError(
            "Student forward failed. Attempts -> "
            + " | ".join(errors)
        )


# ---------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------

class ClipPreprocessor:
    """
    BGR OpenCV frames -> [1,C,T,H,W].

    Default:
        resize -> RGB -> float/255

    Optional:
        channel-wise mean/std normalization
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.image_size = int(image_size)

        if (mean is None) != (std is None):
            raise ValueError(
                "--mean and --std must be supplied together."
            )

        if mean is not None:
            if len(mean) != 3 or len(std) != 3:
                raise ValueError(
                    "mean/std must each contain exactly 3 values."
                )

            self.mean = torch.tensor(
                mean,
                dtype=torch.float32,
            ).view(3, 1, 1, 1)

            self.std = torch.tensor(
                std,
                dtype=torch.float32,
            ).view(3, 1, 1, 1)

            if torch.any(self.std <= 0):
                raise ValueError(
                    "All std values must be > 0."
                )

        else:
            self.mean = None
            self.std = None

    def frame_to_rgb(
        self,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        resized = cv2.resize(
            frame_bgr,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        return rgb

    def __call__(
        self,
        frames_bgr: Sequence[np.ndarray],
    ) -> torch.Tensor:
        if not frames_bgr:
            raise ValueError("Cannot preprocess an empty frame list.")

        rgb_frames = [
            self.frame_to_rgb(frame)
            for frame in frames_bgr
        ]

        array = np.stack(
            rgb_frames,
            axis=0,
        )  # T,H,W,C

        clip = torch.from_numpy(
            array,
        ).permute(
            3, 0, 1, 2,
        ).contiguous().float() / 255.0

        if self.mean is not None:
            clip = (
                clip - self.mean
            ) / self.std

        return clip.unsqueeze(0)


# ---------------------------------------------------------------------
# RMTD-FD online alarm state machine
# ---------------------------------------------------------------------

class OnlineFallAlarm:
    """
    Implements the paper's causal smoothing + threshold + persistence rule.

    State:
        recent_probs: last M window probabilities
        high_streak: number of consecutive smoothed probabilities >= theta
        active: whether one merged alarm event is currently active
    """

    def __init__(
        self,
        moving_average_window: int = 3,
        consecutive_windows: int = 2,
        threshold: float = 0.65,
    ) -> None:
        if moving_average_window <= 0:
            raise ValueError("M must be > 0.")

        if consecutive_windows <= 0:
            raise ValueError("C must be > 0.")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                "Alarm threshold theta must be in [0,1]."
            )

        self.M = int(moving_average_window)
        self.C = int(consecutive_windows)
        self.theta = float(threshold)

        self.recent_probs: Deque[float] = deque(
            maxlen=self.M,
        )

        self.high_streak = 0
        self.active = False

        self.event_counter = 0
        self.current_event: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.recent_probs.clear()
        self.high_streak = 0
        self.active = False
        self.event_counter = 0
        self.current_event = None

    def update(
        self,
        raw_probability: float,
        window_id: int,
        frame_index: int,
        time_seconds: float,
    ) -> Dict[str, Any]:
        """
        Consume one window-level fall probability.

        Returns a state dictionary with:
            smoothed_probability
            threshold_met
            high_streak
            alarm_active
            alarm_triggered_now
            event_closed_now
            event_id
        """
        raw_probability = float(raw_probability)

        self.recent_probs.append(
            raw_probability
        )

        smoothed = float(
            sum(self.recent_probs)
            / len(self.recent_probs)
        )

        threshold_met = (
            smoothed >= self.theta
        )

        alarm_triggered_now = False
        event_closed_now = False
        closed_event = None

        if threshold_met:
            self.high_streak += 1

            if (
                not self.active
                and self.high_streak >= self.C
            ):
                self.active = True
                self.event_counter += 1
                alarm_triggered_now = True

                self.current_event = {
                    "event_id": self.event_counter,
                    "trigger_window_id": int(window_id),
                    "trigger_frame_index": int(frame_index),
                    "trigger_time_seconds": float(time_seconds),
                    "trigger_raw_probability": raw_probability,
                    "trigger_smoothed_probability": smoothed,
                    "M": self.M,
                    "C": self.C,
                    "theta": self.theta,
                }

        else:
            self.high_streak = 0

            if self.active:
                self.active = False
                event_closed_now = True

                if self.current_event is not None:
                    self.current_event.update({
                        "end_window_id": int(window_id - 1),
                        "end_frame_index": int(frame_index),
                        "end_time_seconds": float(time_seconds),
                    })

                    closed_event = dict(
                        self.current_event
                    )

                self.current_event = None

        return {
            "raw_probability": raw_probability,
            "smoothed_probability": smoothed,
            "threshold_met": bool(threshold_met),
            "high_streak": int(self.high_streak),
            "alarm_active": bool(self.active),
            "alarm_triggered_now": bool(alarm_triggered_now),
            "event_closed_now": bool(event_closed_now),
            "event_id": (
                int(self.event_counter)
                if self.active
                else -1
            ),
            "closed_event": closed_event,
        }

    def close_at_end(
        self,
        last_window_id: int,
        last_frame_index: int,
        last_time_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Close an active merged alarm event at end-of-stream.
        """
        if (
            not self.active
            or self.current_event is None
        ):
            return None

        self.current_event.update({
            "end_window_id": int(last_window_id),
            "end_frame_index": int(last_frame_index),
            "end_time_seconds": float(last_time_seconds),
        })

        event = dict(
            self.current_event
        )

        self.active = False
        self.current_event = None
        self.high_streak = 0

        return event


# ---------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------

def parse_source(
    raw: str,
) -> Union[int, str]:
    """
    '0' -> camera index 0
    '/path/video.mp4' -> path string
    """
    stripped = raw.strip()

    if re_full_int(stripped):
        return int(stripped)

    return stripped


def re_full_int(text: str) -> bool:
    if text.startswith(("+", "-")):
        return text[1:].isdigit()
    return text.isdigit()


def open_capture(
    source: Union[int, str],
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video/camera source: {source}"
        )

    return cap


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def draw_status(
    frame: np.ndarray,
    raw_prob: Optional[float],
    smoothed_prob: Optional[float],
    alarm_active: bool,
    high_streak: int,
    threshold: float,
    frame_index: int,
    window_id: int,
) -> np.ndarray:
    out = frame.copy()

    h, w = out.shape[:2]

    # Dark information panel.
    panel_h = min(150, h)
    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(w, 620), panel_h),
        (0, 0, 0),
        -1,
    )

    out = cv2.addWeighted(
        overlay,
        0.55,
        out,
        0.45,
        0,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        out,
        f"Frame: {frame_index}   Window: {window_id}",
        (15, 28),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    raw_text = (
        "Raw fall prob: --"
        if raw_prob is None
        else f"Raw fall prob: {raw_prob:.3f}"
    )

    smooth_text = (
        "Smoothed prob: --"
        if smoothed_prob is None
        else f"Smoothed prob: {smoothed_prob:.3f}"
    )

    cv2.putText(
        out,
        raw_text,
        (15, 58),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        smooth_text,
        (15, 88),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"theta={threshold:.2f}  streak={high_streak}",
        (15, 118),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if alarm_active:
        cv2.putText(
            out,
            "FALL ALARM",
            (max(15, w - 250), 42),
            font,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            out,
            (3, 3),
            (w - 4, h - 4),
            (0, 0, 255),
            5,
        )

    return out


# ---------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------

@torch.inference_mode()
def run_inference(
    source: Union[int, str],
    model: nn.Module,
    caller: ModelCaller,
    device: torch.device,
    preprocessor: ClipPreprocessor,
    clip_len: int,
    stride: int,
    fall_class_index: int,
    alarm: OnlineFallAlarm,
    use_amp: bool,
    display: bool,
    save_video: bool,
    output_video_path: Path,
    output_fps: Optional[float],
    max_frames: Optional[int],
) -> Dict[str, Any]:

    if clip_len <= 0:
        raise ValueError("clip_len must be > 0.")

    if stride <= 0:
        raise ValueError("stride must be > 0.")

    cap = open_capture(source)

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if (
        not math.isfinite(source_fps)
        or source_fps <= 0
    ):
        source_fps = 30.0
        fps_is_fallback = True
    else:
        fps_is_fallback = False

    writer = None

    frame_buffer: Deque[np.ndarray] = deque(
        maxlen=clip_len,
    )

    window_rows: List[Dict[str, Any]] = []
    alarm_events: List[Dict[str, Any]] = []

    frame_index = -1
    window_id = -1
    frames_since_last_window = 0

    latest_raw: Optional[float] = None
    latest_smooth: Optional[float] = None
    latest_alarm_active = False
    latest_streak = 0

    infer_time_sum = 0.0
    infer_calls = 0

    stopped_by_user = False

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            if (
                max_frames is not None
                and frame_index >= max_frames
            ):
                break

            frame_buffer.append(
                frame.copy()
            )

            should_infer = False

            if len(frame_buffer) == clip_len:
                if window_id < 0:
                    should_infer = True
                    frames_since_last_window = 0

                else:
                    frames_since_last_window += 1

                    if frames_since_last_window >= stride:
                        should_infer = True
                        frames_since_last_window = 0

            if should_infer:
                window_id += 1

                clip = preprocessor(
                    list(frame_buffer)
                ).to(
                    device,
                    non_blocking=True,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                start_infer = time.perf_counter()

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    logits = caller(
                        clip
                    )

                probs = F.softmax(
                    logits.float(),
                    dim=1,
                )

                fall_prob = float(
                    probs[0, fall_class_index].item()
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                elapsed = (
                    time.perf_counter()
                    - start_infer
                )

                infer_time_sum += elapsed
                infer_calls += 1

                time_seconds = (
                    frame_index / source_fps
                )

                alarm_state = alarm.update(
                    raw_probability=fall_prob,
                    window_id=window_id,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                )

                latest_raw = fall_prob
                latest_smooth = float(
                    alarm_state[
                        "smoothed_probability"
                    ]
                )

                latest_alarm_active = bool(
                    alarm_state[
                        "alarm_active"
                    ]
                )

                latest_streak = int(
                    alarm_state[
                        "high_streak"
                    ]
                )

                row = {
                    "window_id": window_id,
                    "frame_index": frame_index,
                    "time_seconds": time_seconds,
                    "raw_fall_probability": fall_prob,
                    "smoothed_fall_probability": latest_smooth,
                    "threshold_met": int(
                        alarm_state[
                            "threshold_met"
                        ]
                    ),
                    "high_streak": latest_streak,
                    "alarm_active": int(
                        latest_alarm_active
                    ),
                    "alarm_triggered_now": int(
                        alarm_state[
                            "alarm_triggered_now"
                        ]
                    ),
                    "event_id": int(
                        alarm_state[
                            "event_id"
                        ]
                    ),
                    "inference_seconds": elapsed,
                }

                window_rows.append(
                    row
                )

                if (
                    alarm_state[
                        "alarm_triggered_now"
                    ]
                ):
                    print(
                        f"[ALARM] event={alarm.event_counter} "
                        f"window={window_id} "
                        f"time={time_seconds:.3f}s "
                        f"raw={fall_prob:.4f} "
                        f"smooth={latest_smooth:.4f}"
                    )

                closed_event = alarm_state.get(
                    "closed_event"
                )

                if closed_event is not None:
                    alarm_events.append(
                        closed_event
                    )

            if display or save_video:
                annotated = draw_status(
                    frame=frame,
                    raw_prob=latest_raw,
                    smoothed_prob=latest_smooth,
                    alarm_active=latest_alarm_active,
                    high_streak=latest_streak,
                    threshold=alarm.theta,
                    frame_index=frame_index,
                    window_id=max(window_id, 0),
                )

                if save_video:
                    if writer is None:
                        h, w = annotated.shape[:2]

                        save_fps = (
                            float(output_fps)
                            if output_fps is not None
                            else source_fps
                        )

                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            str(output_video_path),
                            fourcc,
                            save_fps,
                            (w, h),
                        )

                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Cannot open output video: "
                                f"{output_video_path}"
                            )

                    writer.write(
                        annotated
                    )

                if display:
                    cv2.imshow(
                        "RMTD-FD Online Inference",
                        annotated,
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key in (
                        ord("q"),
                        27,
                    ):
                        stopped_by_user = True
                        break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if display:
            cv2.destroyAllWindows()

    last_time_seconds = (
        max(frame_index, 0)
        / source_fps
    )

    final_event = alarm.close_at_end(
        last_window_id=max(window_id, 0),
        last_frame_index=max(frame_index, 0),
        last_time_seconds=last_time_seconds,
    )

    if final_event is not None:
        alarm_events.append(
            final_event
        )

    avg_inference_seconds = (
        infer_time_sum
        / max(infer_calls, 1)
    )

    windows_per_second = (
        1.0 / avg_inference_seconds
        if avg_inference_seconds > 0
        else float("nan")
    )

    return {
        "window_rows": window_rows,
        "alarm_events": alarm_events,
        "source_fps": source_fps,
        "source_fps_was_fallback": fps_is_fallback,
        "frames_processed": max(frame_index + 1, 0),
        "windows_processed": len(window_rows),
        "alarm_event_count": len(alarm_events),
        "average_model_inference_seconds_per_window": avg_inference_seconds,
        "model_windows_per_second": windows_per_second,
        "stopped_by_user": stopped_by_user,
    }


# ---------------------------------------------------------------------
# Smoke-test model
# ---------------------------------------------------------------------

class TinyStudent(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            3,
            4,
            kernel_size=3,
            padding=1,
        )

        self.head = nn.Linear(
            4,
            num_classes,
        )

    def forward(
        self,
        x: Any,
    ) -> torch.Tensor:
        if isinstance(
            x,
            (list, tuple),
        ):
            x = x[0]

        x = F.gelu(
            self.conv(x)
        )

        x = x.mean(
            dim=(2, 3, 4)
        )

        return self.head(x)


def build_smoke_video(
    path: Path,
    fps: int = 20,
    frames: int = 40,
) -> None:
    h, w = 48, 64

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Smoke test could not create video."
        )

    try:
        for i in range(frames):
            img = np.zeros(
                (h, w, 3),
                dtype=np.uint8,
            )

            value = int(
                (i / max(frames - 1, 1))
                * 255
            )

            img[:, :] = (
                value,
                value,
                value,
            )

            writer.write(
                img
            )

    finally:
        writer.release()


def run_smoke_test() -> None:
    print(
        "[Smoke test] RMTD-FD inference / online alarm"
    )

    set_seed(123)

    device = torch.device(
        "cpu"
    )

    # First directly validate the exact alarm-state logic with deterministic
    # probabilities.
    alarm = OnlineFallAlarm(
        moving_average_window=3,
        consecutive_windows=2,
        threshold=0.65,
    )

    trace = [
        0.20,
        0.40,
        0.90,
        0.95,
        0.98,
        0.10,
        0.10,
    ]

    events = []

    for i, p in enumerate(trace):
        state = alarm.update(
            raw_probability=p,
            window_id=i,
            frame_index=i * 8,
            time_seconds=i * 0.4,
        )

        if state["closed_event"] is not None:
            events.append(
                state["closed_event"]
            )

    end_event = alarm.close_at_end(
        last_window_id=len(trace) - 1,
        last_frame_index=(len(trace) - 1) * 8,
        last_time_seconds=(len(trace) - 1) * 0.4,
    )

    if end_event is not None:
        events.append(end_event)

    assert len(events) == 1, events
    assert events[0]["event_id"] == 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        video = root / "smoke.mp4"

        build_smoke_video(
            video,
            fps=20,
            frames=40,
        )

        model = TinyStudent().to(
            device
        )

        caller = ModelCaller(
            model,
            input_style="auto",
            num_classes=2,
        )

        preprocessor = ClipPreprocessor(
            image_size=32,
        )

        alarm2 = OnlineFallAlarm(
            moving_average_window=3,
            consecutive_windows=2,
            threshold=0.65,
        )

        result = run_inference(
            source=str(video),
            model=model,
            caller=caller,
            device=device,
            preprocessor=preprocessor,
            clip_len=8,
            stride=4,
            fall_class_index=1,
            alarm=alarm2,
            use_amp=False,
            display=False,
            save_video=False,
            output_video_path=root / "unused.mp4",
            output_fps=None,
            max_frames=None,
        )

        assert result[
            "frames_processed"
        ] == 40

        assert result[
            "windows_processed"
        ] > 0

        assert len(
            result["window_rows"]
        ) == result[
            "windows_processed"
        ]

        print(
            "[Smoke test] PASS | "
            f"frames={result['frames_processed']} "
            f"windows={result['windows_processed']} "
            f"events={result['alarm_event_count']}"
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "RMTD-FD student-only online inference "
            "and fall-alarm pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
    )

    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video path or camera index such as 0.",
    )

    p.add_argument(
        "--model-factory",
        type=str,
        default=None,
        help=(
            "UniFormer-XS builder as "
            "module:function or file.py:function."
        ),
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    p.add_argument(
        "--input-style",
        choices=("auto", "tensor", "list"),
        default="auto",
    )

    p.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
    )

    # Paper input protocol.
    p.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="L in the paper.",
    )

    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help="s in the paper.",
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    p.add_argument(
        "--fall-class-index",
        type=int,
        choices=(0, 1),
        default=1,
    )

    # Paper alarm protocol.
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

    # Optional preprocessing normalization.
    p.add_argument(
        "--mean",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help=(
            "Optional training-time RGB mean. "
            "Not invented by default."
        ),
    )

    p.add_argument(
        "--std",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help=(
            "Optional training-time RGB std. "
            "Must be supplied together with --mean."
        ),
    )

    # Runtime.
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    p.add_argument(
        "--amp",
        action="store_true",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--display",
        action="store_true",
        help="Show live annotated frames; press q or Esc to stop.",
    )

    p.add_argument(
        "--save-video",
        action="store_true",
    )

    p.add_argument(
        "--output-fps",
        type=float,
        default=None,
    )

    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debugging limit.",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default="./runs/rmtd_fd_online",
    )

    return p


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.smoke_test:
        return

    required = {
        "--source": args.source,
        "--model-factory": args.model_factory,
        "--checkpoint": args.checkpoint,
    }

    missing = [
        name
        for name, value in required.items()
        if value is None or value == ""
    ]

    if missing:
        raise SystemExit(
            "Missing required argument(s): "
            + ", ".join(missing)
        )

    if args.clip_len <= 0:
        raise SystemExit(
            "--clip-len must be > 0."
        )

    if args.stride <= 0:
        raise SystemExit(
            "--stride must be > 0."
        )

    if args.moving_average_window <= 0:
        raise SystemExit(
            "--moving-average-window must be > 0."
        )

    if args.consecutive_windows <= 0:
        raise SystemExit(
            "--consecutive-windows must be > 0."
        )

    if not (
        0.0
        <= args.alarm_threshold
        <= 1.0
    ):
        raise SystemExit(
            "--alarm-threshold must be in [0,1]."
        )

    if (
        args.mean is None
    ) != (
        args.std is None
    ):
        raise SystemExit(
            "--mean and --std must be provided together."
        )


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

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise SystemExit(
            "CUDA was requested but is not available."
        )

    use_amp = bool(
        args.amp
        and device.type == "cuda"
    )

    source = parse_source(
        args.source
    )

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_dir
        / "annotated_output.mp4"
    )

    print("=" * 76)
    print("RMTD-FD Inference / Online Alarm")
    print("Inference model : UniFormer-XS student only")
    print(f"Input source    : {source}")
    print(f"Device          : {device}")
    print(
        f"Sliding window  : "
        f"L={args.clip_len}, stride={args.stride}"
    )
    print(
        f"Alarm protocol  : "
        f"M={args.moving_average_window}, "
        f"C={args.consecutive_windows}, "
        f"theta={args.alarm_threshold}"
    )
    print("=" * 76)

    model = build_model(
        args.model_factory,
        num_classes=2,
    ).to(
        device
    )

    checkpoint_metadata = load_student_checkpoint(
        model,
        args.checkpoint,
        strict=not args.non_strict_checkpoint,
    )

    model.eval()

    caller = ModelCaller(
        model,
        input_style=args.input_style,
        num_classes=2,
    )

    preprocessor = ClipPreprocessor(
        image_size=args.image_size,
        mean=args.mean,
        std=args.std,
    )

    alarm = OnlineFallAlarm(
        moving_average_window=args.moving_average_window,
        consecutive_windows=args.consecutive_windows,
        threshold=args.alarm_threshold,
    )

    result = run_inference(
        source=source,
        model=model,
        caller=caller,
        device=device,
        preprocessor=preprocessor,
        clip_len=args.clip_len,
        stride=args.stride,
        fall_class_index=args.fall_class_index,
        alarm=alarm,
        use_amp=use_amp,
        display=args.display,
        save_video=args.save_video,
        output_video_path=output_video,
        output_fps=args.output_fps,
        max_frames=args.max_frames,
    )

    window_path = (
        output_dir
        / "window_predictions.csv"
    )

    alarm_path = (
        output_dir
        / "alarm_events.csv"
    )

    write_csv(
        window_path,
        result["window_rows"],
    )

    write_csv(
        alarm_path,
        result["alarm_events"],
    )

    summary = {
        "source": str(source),
        "student_only_inference": True,
        "checkpoint": str(
            Path(
                args.checkpoint
            ).expanduser().resolve()
        ),
        "checkpoint_metadata": checkpoint_metadata,
        "paper_protocol": {
            "clip_length_L": args.clip_len,
            "stride_s": args.stride,
            "input_size": args.image_size,
            "moving_average_M": args.moving_average_window,
            "persistence_C": args.consecutive_windows,
            "threshold_theta": args.alarm_threshold,
            "fall_class_index": args.fall_class_index,
        },
        "preprocessing": {
            "rgb_resize_and_scale_0_1": True,
            "mean": args.mean,
            "std": args.std,
        },
        "result": {
            k: v
            for k, v in result.items()
            if k not in (
                "window_rows",
                "alarm_events",
            )
        },
        "alarm_events": result[
            "alarm_events"
        ],
        "args": vars(args),
    }

    summary_path = (
        output_dir
        / "inference_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nInference complete.")
    print(
        f"Frames processed : "
        f"{result['frames_processed']}"
    )
    print(
        f"Windows processed: "
        f"{result['windows_processed']}"
    )
    print(
        f"Alarm events     : "
        f"{result['alarm_event_count']}"
    )
    print(
        f"Avg model time   : "
        f"{result['average_model_inference_seconds_per_window']*1000:.3f} ms/window"
    )
    print(
        f"Model throughput : "
        f"{result['model_windows_per_second']:.2f} windows/s"
    )

    print("\nSaved:")
    print(" ", window_path)
    print(" ", alarm_path)
    print(" ", summary_path)

    if args.save_video:
        print(" ", output_video)


if __name__ == "__main__":
    main()
