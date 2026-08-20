#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess UP-Fall for the RMTD-FD pipeline
===========================================

What this script produces
-------------------------
- 16-frame RGB clips with stride 8
- resized to 224x224
- binary labels: fall / non-fall
- full-frame tensors for T1/T2/student
- human-ROI tensors for T3 (YOLOv8n person detector, confidence 0.25)
- train.csv / val.csv / test.csv compatible with pretrain_rmtd_fd_teachers.py
  and train_rmtd_fd_distill.py

Paper protocol implemented
--------------------------
- split BEFORE clip generation
- fixed cross-subject split with 12 / 2 / 3 subjects
- sliding window: 16 frames, stride 8
- output resolution: 224x224
- T3 ROI preprocessing:
    * COCO-pretrained YOLOv8n
    * person class only
    * confidence threshold 0.25
    * if only one person is detected: highest-confidence person
    * if multiple persons are present: choose the track that appears in the most
      frames, then the one with the largest mean box area
    * if no selected detection is available in a frame: fall back to full frame

Important reproducibility note
------------------------------
The manuscript reports a 12/2/3 cross-subject split but does not list the exact
subject IDs. Therefore:
- if --split-json is supplied, this script uses it exactly;
- otherwise it creates ONE deterministic 12/2/3 split from the discovered
  subjects using --split-seed, writes split_used.json, and prints a warning.
Use your experiment's actual subject IDs in --split-json if you have them.

UP-Fall raw-image packaging can differ depending on how Camera1/Camera2 archives
were downloaded/extracted. The script supports:
- zip archives containing RGB images
- extracted image-sequence directories
- common video files

For the most reliable run, use --source-index if auto-discovery cannot infer
subject/activity/trial/camera. Source index CSV columns:
    source_path,subject,activity,trial,camera

Example:
    python preprocess_upfall_rmtd_fd.py \
        --data-root /data/UP-Fall \
        --output-root /data/RMTD_FD/UP_Fall \
        --camera 1 \
        --split-json upfall_split.json

split JSON:
{
  "train": [1,2,3,4,5,6,7,8,9,10,11,12],
  "val": [13,14],
  "test": [15,16,17]
}

The numbers above are ONLY an example format. Do not treat them as the paper's
unreported exact split.

Dependencies:
    pip install torch numpy opencv-python
    pip install ultralytics   # required unless --skip-roi

Quick self-test:
    python preprocess_upfall_rmtd_fd.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ARCHIVE_EXTS = {".zip"}

# UP-Fall contains five simulated fall activities and six ADLs.
# Activity IDs 1..5 are treated as fall; 6..11 as non-fall.
FALL_ACTIVITY_IDS = {1, 2, 3, 4, 5}


@dataclass(frozen=True)
class SequenceInfo:
    source: Path
    subject: int
    activity: int
    trial: int
    camera: int

    @property
    def sequence_id(self) -> str:
        return (
            f"S{self.subject:02d}_A{self.activity:02d}_"
            f"T{self.trial:02d}_C{self.camera}"
        )

    @property
    def label(self) -> int:
        return int(self.activity in FALL_ACTIVITY_IDS)


def natural_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def read_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV failed to decode an image.")
    return frame


def iter_frames_from_source(source: Path) -> Iterator[np.ndarray]:
    if source.is_dir():
        files = sorted(
            [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: natural_key(str(p.relative_to(source))),
        )
        if not files:
            raise RuntimeError(f"No images found in directory: {source}")
        for p in files:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame
        return

    suffix = source.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(source, "r") as zf:
            names = sorted(
                [
                    n for n in zf.namelist()
                    if Path(n).suffix.lower() in IMAGE_EXTS
                    and not n.endswith("/")
                ],
                key=natural_key,
            )
            if not names:
                raise RuntimeError(f"No RGB images found in zip: {source}")
            for name in names:
                yield read_image_bytes(zf.read(name))
        return

    if suffix in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        return

    raise ValueError(f"Unsupported sequence source: {source}")


def _named_int(path_text: str, names: Sequence[str]) -> Optional[int]:
    joined = "|".join(re.escape(x) for x in names)
    patterns = [
        rf"(?:^|[\\/_.\-\s])(?:{joined})[ _.\-]*0*(\d+)(?=$|[\\/_.\-\s])",
        rf"(?:{joined})[ _.\-]*0*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, path_text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def infer_subject_activity_trial(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    text = str(path)

    # First prefer descriptive tokens. Do NOT search bare s/a/t over the whole
    # absolute path because temporary/user directory names may contain them.
    subject = _named_int(text, ("subject", "subj", "sub"))
    activity = _named_int(text, ("activity", "act"))
    trial = _named_int(text, ("trial", "attempt", "try"))

    # Then accept compact directory tokens such as S01/A03/T02 only when the
    # complete path component matches that token.
    for part in path.parts:
        m = re.fullmatch(r"[sS]0*(\d+)", part)
        if m and subject is None:
            subject = int(m.group(1))
        m = re.fullmatch(r"[aA]0*(\d+)", part)
        if m and activity is None:
            activity = int(m.group(1))
        m = re.fullmatch(r"[tT]0*(\d+)", part)
        if m and trial is None:
            trial = int(m.group(1))

    if subject is not None and activity is not None and trial is not None:
        return subject, activity, trial

    # Official organization is subject/activity/trial. If directories are pure
    # integers, use the last three numeric directory components.
    numeric_dirs = []
    for part in path.parent.parts:
        if re.fullmatch(r"0*\d+", part):
            numeric_dirs.append(int(part))
    if len(numeric_dirs) >= 3:
        s, a, t = numeric_dirs[-3:]
        subject = subject if subject is not None else s
        activity = activity if activity is not None else a
        trial = trial if trial is not None else t
    return subject, activity, trial


def camera_matches(name: str, camera: int) -> bool:
    text = name.lower()
    aliases = [
        f"camera{camera}",
        f"camera_{camera}",
        f"camera-{camera}",
        f"cam{camera}",
        f"cam_{camera}",
        f"cam-{camera}",
    ]
    return any(a in text for a in aliases)


def discover_sources(data_root: Path, camera: int) -> List[SequenceInfo]:
    candidates: List[Path] = []

    # Archives/videos with explicit camera marker.
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (ARCHIVE_EXTS | VIDEO_EXTS):
            if camera_matches(p.name, camera) or camera_matches(str(p.parent), camera):
                candidates.append(p)

    # Extracted camera directories. Only take directories with direct image files
    # to avoid adding every ancestor.
    for d in data_root.rglob("*"):
        if not d.is_dir():
            continue
        if not (camera_matches(d.name, camera) or camera_matches(str(d.parent), camera)):
            continue
        try:
            has_images = any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTS
                for p in d.iterdir()
            )
        except OSError:
            has_images = False
        if has_images:
            candidates.append(d)

    infos: Dict[Tuple[int, int, int, int], SequenceInfo] = {}
    skipped: List[str] = []

    for source in sorted(set(candidates), key=lambda p: natural_key(str(p))):
        s, a, t = infer_subject_activity_trial(source)
        if s is None or a is None or t is None:
            skipped.append(str(source))
            continue
        if not (1 <= s <= 17 and 1 <= a <= 11 and 1 <= t <= 3):
            skipped.append(str(source))
            continue

        key = (s, a, t, camera)
        # Prefer zip/video over an accidentally duplicated extracted folder only
        # if the key has not already been seen; user can use source-index to override.
        if key not in infos:
            infos[key] = SequenceInfo(source, s, a, t, camera)

    if skipped:
        print(f"[UP-Fall] Auto-discovery skipped {len(skipped)} ambiguous paths.")
        print("  Use --source-index if required. Examples:")
        for x in skipped[:5]:
            print("   ", x)

    return sorted(
        infos.values(),
        key=lambda x: (x.subject, x.activity, x.trial, x.camera),
    )


def load_source_index(csv_path: Path, data_root: Path, camera: int) -> List[SequenceInfo]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"source_path", "subject", "activity", "trial", "camera"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Source index must contain columns: {sorted(required)}"
        )

    out = []
    for row in rows:
        if int(row["camera"]) != camera:
            continue
        p = Path(row["source_path"]).expanduser()
        if not p.is_absolute():
            p = data_root / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        out.append(
            SequenceInfo(
                source=p,
                subject=int(row["subject"]),
                activity=int(row["activity"]),
                trial=int(row["trial"]),
                camera=int(row["camera"]),
            )
        )
    return out


def save_discovered_sources(path: Path, infos: Sequence[SequenceInfo], root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "subject", "activity", "trial", "camera", "label"],
        )
        writer.writeheader()
        for x in infos:
            try:
                rel = x.source.relative_to(root)
                source_text = str(rel)
            except ValueError:
                source_text = str(x.source)
            writer.writerow({
                "source_path": source_text,
                "subject": x.subject,
                "activity": x.activity,
                "trial": x.trial,
                "camera": x.camera,
                "label": x.label,
            })


def load_or_make_split(
    infos: Sequence[SequenceInfo],
    split_json: Optional[Path],
    seed: int,
) -> Dict[str, List[int]]:
    subjects = sorted({x.subject for x in infos})
    if split_json is not None:
        with split_json.open("r", encoding="utf-8") as f:
            split = json.load(f)
        for key, n in (("train", 12), ("val", 2), ("test", 3)):
            if key not in split:
                raise ValueError(f"split JSON missing key: {key}")
            split[key] = [int(x) for x in split[key]]
            if len(split[key]) != n:
                raise ValueError(
                    f"UP-Fall paper protocol requires {n} {key} subjects; "
                    f"got {len(split[key])}."
                )
    else:
        if len(subjects) != 17:
            raise ValueError(
                f"Expected 17 discovered subjects before generating a 12/2/3 split, "
                f"but found {len(subjects)}: {subjects}. "
                "Use --source-index or --split-json after checking your download."
            )
        rng = random.Random(seed)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        split = {
            "train": sorted(shuffled[:12]),
            "val": sorted(shuffled[12:14]),
            "test": sorted(shuffled[14:17]),
        }
        print(
            "[WARNING] The paper does not report exact UP-Fall subject IDs. "
            f"Generated a deterministic split with seed={seed}: {split}"
        )

    all_ids = split["train"] + split["val"] + split["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Subject split overlaps across train/val/test.")
    if set(all_ids) != set(subjects):
        raise ValueError(
            f"Split subject IDs {sorted(set(all_ids))} do not match discovered "
            f"subjects {subjects}."
        )
    return split


class PersonDetector:
    def __init__(self, weights: str, conf: float, device: Optional[str]):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "ROI generation requires `ultralytics`. Install it with "
                "`pip install ultralytics`, or use --skip-roi only for debugging."
            ) from exc
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.device = device

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        kwargs = dict(
            source=frame_bgr,
            classes=[0],
            conf=self.conf,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        boxes = []
        if result.boxes is None:
            return boxes
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = map(float, b.tolist())
            boxes.append((x1, y1, x2, y2, float(c)))
        return boxes


def box_area(box) -> float:
    x1, y1, x2, y2, _ = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def select_subject_boxes(
    detections: Sequence[List[Tuple[float, float, float, float, float]]],
    iou_threshold: float = 0.20,
) -> List[Optional[Tuple[float, float, float, float, float]]]:
    """
    Paper-inspired operationalization of:
    'multi-person: the subject that appears most consistently across frames and
    has the largest average bounding-box area.'

    We greedily link person detections by IoU, rank tracks by frame count and
    then average area. A single-person frame naturally contributes its only box.
    """
    tracks: List[Dict] = []

    for frame_idx, boxes in enumerate(detections):
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        assigned_tracks = set()

        for box in boxes:
            best_track = None
            best_score = -1.0
            for ti, tr in enumerate(tracks):
                if ti in assigned_tracks:
                    continue
                if tr["last_frame"] != frame_idx - 1:
                    continue
                score = iou(tr["last_box"], box)
                if score > best_score:
                    best_score = score
                    best_track = ti

            if best_track is not None and best_score >= iou_threshold:
                tr = tracks[best_track]
                tr["boxes"][frame_idx] = box
                tr["last_box"] = box
                tr["last_frame"] = frame_idx
                assigned_tracks.add(best_track)
            else:
                tracks.append({
                    "boxes": {frame_idx: box},
                    "last_box": box,
                    "last_frame": frame_idx,
                })
                assigned_tracks.add(len(tracks) - 1)

    if not tracks:
        return [None] * len(detections)

    def rank(tr):
        boxes = list(tr["boxes"].values())
        return (len(boxes), float(np.mean([box_area(b) for b in boxes])))

    best = max(tracks, key=rank)
    selected = []
    for i, frame_boxes in enumerate(detections):
        if i in best["boxes"]:
            selected.append(best["boxes"][i])
        elif len(frame_boxes) == 1:
            # If tracking temporarily broke but only one person is visible,
            # use the highest-confidence (only) detection.
            selected.append(frame_boxes[0])
        else:
            selected.append(None)
    return selected


def crop_box(frame: np.ndarray, box) -> Optional[np.ndarray]:
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2, _ = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def frame_to_rgb224(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def stack_uint8_clip(frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames_rgb, axis=0)  # T,H,W,C
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # C,T,H,W


def process_sequence(
    info: SequenceInfo,
    split_name: str,
    output_root: Path,
    clip_len: int,
    stride: int,
    image_size: int,
    detector: Optional[PersonDetector],
) -> List[Dict[str, str]]:
    full_dir = output_root / "clips" / split_name
    roi_dir = output_root / "rois" / split_name
    full_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    frames_q: deque = deque()
    dets_q: deque = deque()
    rows: List[Dict[str, str]] = []
    window_id = 0

    for frame_idx, frame in enumerate(iter_frames_from_source(info.source)):
        frames_q.append(frame)
        dets_q.append(detector.detect(frame) if detector else [])

        if len(frames_q) < clip_len:
            continue

        if len(frames_q) == clip_len:
            orig_frames = list(frames_q)
            full_rgb = [frame_to_rgb224(f, image_size) for f in orig_frames]

            if detector is not None:
                selected = select_subject_boxes(list(dets_q))
                roi_rgb = []
                for f, box in zip(orig_frames, selected):
                    crop = crop_box(f, box)
                    if crop is None:
                        crop = f  # paper fallback when detection fails
                    roi_rgb.append(frame_to_rgb224(crop, image_size))
            else:
                roi_rgb = full_rgb

            start_frame = window_id * stride
            stem = f"{info.sequence_id}_W{window_id:05d}_F{start_frame:06d}"
            full_path = full_dir / f"{stem}.pt"
            roi_path = roi_dir / f"{stem}.pt"

            torch.save(stack_uint8_clip(full_rgb), full_path)
            torch.save(stack_uint8_clip(roi_rgb), roi_path)

            rows.append({
                "clip_path": str(full_path.relative_to(output_root)),
                "roi_path": str(roi_path.relative_to(output_root)),
                "label": str(info.label),
                "dataset": "UP-Fall",
                "split": split_name,
                "subject": str(info.subject),
                "activity": str(info.activity),
                "trial": str(info.trial),
                "camera": str(info.camera),
                "sequence_id": info.sequence_id,
                "window_id": str(window_id),
                "start_frame": str(start_frame),
            })
            window_id += 1

            # slide by exactly `stride` frames
            for _ in range(min(stride, len(frames_q))):
                frames_q.popleft()
                dets_q.popleft()

    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = [
        "clip_path", "roi_path", "label", "dataset", "split",
        "subject", "activity", "trial", "camera",
        "sequence_id", "window_id", "start_frame",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_preprocess(args) -> Dict[str, int]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.source_index:
        infos = load_source_index(
            Path(args.source_index).expanduser().resolve(),
            data_root,
            args.camera,
        )
    else:
        infos = discover_sources(data_root, args.camera)

    if not infos:
        raise RuntimeError(
            "No UP-Fall RGB sequences discovered. Check --data-root/--camera "
            "or provide --source-index."
        )

    save_discovered_sources(
        output_root / "discovered_sources.csv",
        infos,
        data_root,
    )

    split = load_or_make_split(
        infos,
        Path(args.split_json).expanduser().resolve() if args.split_json else None,
        args.split_seed,
    )
    with (output_root / "split_used.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    subject_to_split = {}
    for name in ("train", "val", "test"):
        for s in split[name]:
            subject_to_split[s] = name

    detector = None
    if not args.skip_roi:
        detector = PersonDetector(
            weights=args.yolo_weights,
            conf=args.yolo_conf,
            device=args.yolo_device,
        )
    else:
        print(
            "[WARNING] --skip-roi is active: ROI tensors will be copies of full "
            "frames. This is for debugging, not the paper's T3 preprocessing."
        )

    manifests = {"train": [], "val": [], "test": []}
    print(f"[UP-Fall] sequences discovered: {len(infos)}")
    for idx, info in enumerate(infos, start=1):
        split_name = subject_to_split[info.subject]
        print(
            f"[{idx}/{len(infos)}] {split_name:5s} {info.sequence_id} "
            f"label={info.label} <- {info.source}"
        )
        rows = process_sequence(
            info,
            split_name,
            output_root,
            args.clip_len,
            args.stride,
            args.image_size,
            detector,
        )
        manifests[split_name].extend(rows)

    counts = {}
    for name, rows in manifests.items():
        write_manifest(output_root / f"{name}.csv", rows)
        counts[name] = len(rows)
    counts["total"] = sum(counts.values())

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "UP-Fall",
                "camera": args.camera,
                "clip_len": args.clip_len,
                "stride": args.stride,
                "image_size": args.image_size,
                "split": split,
                "clip_counts": counts,
                "expected_paper_total_clips": 17901,
                "roi_enabled": not args.skip_roi,
                "yolo_conf": args.yolo_conf,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[UP-Fall] clip counts:", counts)
    if counts["total"] != 17901:
        print(
            "[NOTE] The paper reports 17,901 UP-Fall clips. Your generated total "
            f"is {counts['total']}. This can differ if the selected camera, raw "
            "download/extraction, missing frames, or unreported exact split/source "
            "choices differ. Do not force the count by duplicating/deleting samples."
        )
    return counts


def make_synthetic_upfall(root: Path) -> Path:
    # Tiny synthetic structure only for code validation.
    # 17 subjects, one activity/trial each, camera1, 20 frames => one window with 16/8.
    for s in range(1, 18):
        d = root / str(s) / "1" / "1" / "Camera1"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            img = np.full((32, 48, 3), (s * 10 + i) % 255, dtype=np.uint8)
            cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), img)
    return root


def smoke_test() -> None:
    print("[Smoke test] UP-Fall preprocessing")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = make_synthetic_upfall(root / "raw")
        out = root / "out"

        class A:
            data_root = str(data)
            output_root = str(out)
            source_index = None
            camera = 1
            split_json = None
            split_seed = 42
            skip_roi = True
            yolo_weights = "yolov8n.pt"
            yolo_conf = 0.25
            yolo_device = None
            clip_len = 16
            stride = 8
            image_size = 224

        counts = run_preprocess(A())
        assert counts["total"] == 17
        for name in ("train.csv", "val.csv", "test.csv", "split_used.json"):
            assert (out / name).exists(), name
        print("[Smoke test] PASS")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess UP-Fall for RMTD-FD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--data-root", type=str)
    p.add_argument("--output-root", type=str)
    p.add_argument("--source-index", type=str, default=None)
    p.add_argument("--camera", type=int, choices=(1, 2), default=1)
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)

    p.add_argument("--skip-roi", action="store_true")
    p.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-device", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if not args.data_root or not args.output_root:
        raise SystemExit("--data-root and --output-root are required.")
    run_preprocess(args)


if __name__ == "__main__":
    main()
