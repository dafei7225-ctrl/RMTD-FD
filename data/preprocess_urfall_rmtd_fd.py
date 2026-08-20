#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess UR-Fall for the RMTD-FD pipeline
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
- original-sequence split BEFORE clip generation
- fixed stratified sequence-level split: 49 train / 7 val / 14 test
- preserve approximately consistent class proportions
- sliding window: 16 frames, stride 8
- output resolution: 224x224

UR-Fall official RGB organization
---------------------------------
The official dataset has:
- 30 fall sequences
- 40 ADL sequences
- fall RGB sequences for camera 0 and camera 1
- ADL RGB sequences only for camera 0

This script therefore uses camera 0 for all 70 sequences so every class is
processed through the same RGB view. It auto-discovers official-style names:
    fall-01-cam0-rgb.zip ... fall-30-cam0-rgb.zip
    adl-01-cam0-rgb.zip  ... adl-40-cam0-rgb.zip
and also supports extracted directories with the same stems or common videos.

Important reproducibility note
------------------------------
The manuscript reports 49/7/14 fixed stratified sequence-level counts, but does
not list the exact 70 sequence IDs assigned to each split. Therefore:
- if --split-json is supplied, it is used exactly;
- otherwise this script makes a deterministic stratified split using --split-seed
  and writes split_used.json.

The generated split allocates class counts:
    train: 21 falls + 28 ADLs = 49
    val:    3 falls +  4 ADLs = 7
    test:   6 falls +  8 ADLs = 14
which exactly partitions the official 30/40 sequence totals while keeping
approximately the same class ratio in all subsets.

T3 ROI preprocessing
--------------------
Same RMTD-FD protocol:
- COCO-pretrained YOLOv8n
- person class only
- confidence threshold 0.25
- single person: highest-confidence box
- multi-person: track appearing in most frames, then largest average box area
- missing selected detection: full-frame fallback

Example:
    python preprocess_urfall_rmtd_fd.py \
        --data-root /data/UR-Fall \
        --output-root /data/RMTD_FD/UR_Fall

Dependencies:
    pip install torch numpy opencv-python
    pip install ultralytics   # required unless --skip-roi

Quick self-test:
    python preprocess_urfall_rmtd_fd.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class SequenceInfo:
    source: Path
    kind: str  # fall / adl
    number: int

    @property
    def sequence_id(self) -> str:
        return f"{self.kind}-{self.number:02d}"

    @property
    def label(self) -> int:
        return int(self.kind == "fall")


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


UR_PATTERN = re.compile(
    r"^(fall|adl)-(\d+)-cam0-rgb(?:\.(?:zip|mp4|avi|mov|mkv|m4v))?$",
    re.IGNORECASE,
)


def discover_sources(data_root: Path) -> List[SequenceInfo]:
    found: Dict[Tuple[str, int], SequenceInfo] = {}

    for p in data_root.rglob("*"):
        if not (p.is_dir() or p.is_file()):
            continue

        name = p.name
        stem_name = name
        m = UR_PATTERN.match(stem_name)
        if not m:
            continue

        kind = m.group(1).lower()
        number = int(m.group(2))

        if kind == "fall" and not (1 <= number <= 30):
            continue
        if kind == "adl" and not (1 <= number <= 40):
            continue

        # For directories require at least one image.
        if p.is_dir():
            try:
                if not any(
                    q.is_file() and q.suffix.lower() in IMAGE_EXTS
                    for q in p.rglob("*")
                ):
                    continue
            except OSError:
                continue
        elif p.suffix.lower() not in ({".zip"} | VIDEO_EXTS):
            continue

        key = (kind, number)
        if key not in found:
            found[key] = SequenceInfo(p, kind, number)

    return sorted(
        found.values(),
        key=lambda x: (0 if x.kind == "fall" else 1, x.number),
    )


def save_discovered_sources(path: Path, infos: Sequence[SequenceInfo], root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "sequence_id", "kind", "number", "label"],
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
                "sequence_id": x.sequence_id,
                "kind": x.kind,
                "number": x.number,
                "label": x.label,
            })


def make_default_stratified_split(
    infos: Sequence[SequenceInfo],
    seed: int,
) -> Dict[str, List[str]]:
    falls = sorted([x.sequence_id for x in infos if x.kind == "fall"])
    adls = sorted([x.sequence_id for x in infos if x.kind == "adl"])

    if len(falls) != 30 or len(adls) != 40:
        raise ValueError(
            f"Expected official UR-Fall totals 30 fall + 40 ADL sequences, "
            f"found {len(falls)} + {len(adls)}. Check extraction/download."
        )

    rng = random.Random(seed)
    rng.shuffle(falls)
    rng.shuffle(adls)

    split = {
        "train": sorted(falls[:21]) + sorted(adls[:28]),
        "val": sorted(falls[21:24]) + sorted(adls[28:32]),
        "test": sorted(falls[24:30]) + sorted(adls[32:40]),
    }
    print(
        "[WARNING] The paper does not list exact UR-Fall sequence IDs. "
        f"Generated a deterministic stratified split with seed={seed}."
    )
    return split


def load_or_make_split(
    infos: Sequence[SequenceInfo],
    split_json: Optional[Path],
    seed: int,
) -> Dict[str, List[str]]:
    all_ids = {x.sequence_id for x in infos}

    if split_json is not None:
        with split_json.open("r", encoding="utf-8") as f:
            split = json.load(f)
        for key, n in (("train", 49), ("val", 7), ("test", 14)):
            if key not in split:
                raise ValueError(f"split JSON missing key: {key}")
            split[key] = [str(x).lower() for x in split[key]]
            if len(split[key]) != n:
                raise ValueError(
                    f"UR-Fall protocol requires {n} {key} sequences; "
                    f"got {len(split[key])}."
                )
    else:
        split = make_default_stratified_split(infos, seed)

    flat = split["train"] + split["val"] + split["test"]
    if len(flat) != len(set(flat)):
        raise ValueError("Sequence split overlaps across train/val/test.")
    if set(flat) != all_ids:
        missing = sorted(all_ids - set(flat))
        extra = sorted(set(flat) - all_ids)
        raise ValueError(
            f"Split IDs do not match discovered data. Missing={missing}, extra={extra}"
        )

    # Validate approximate class proportions and exact total counts.
    for key, target_fall, target_adl in (
        ("train", 21, 28),
        ("val", 3, 4),
        ("test", 6, 8),
    ):
        fall_n = sum(x.startswith("fall-") for x in split[key])
        adl_n = sum(x.startswith("adl-") for x in split[key])
        if split_json is None and (fall_n != target_fall or adl_n != target_adl):
            raise AssertionError("Internal stratified split construction failed.")
        if split_json is not None:
            print(
                f"[split] {key}: {fall_n} falls + {adl_n} ADLs = {len(split[key])}"
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
        bs = list(tr["boxes"].values())
        return (len(bs), float(np.mean([box_area(b) for b in bs])))

    best = max(tracks, key=rank)
    selected = []
    for i, boxes in enumerate(detections):
        if i in best["boxes"]:
            selected.append(best["boxes"][i])
        elif len(boxes) == 1:
            selected.append(boxes[0])
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
    arr = np.stack(frames_rgb, axis=0)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


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

    for frame in iter_frames_from_source(info.source):
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
                        crop = f
                    roi_rgb.append(frame_to_rgb224(crop, image_size))
            else:
                roi_rgb = full_rgb

            start_frame = window_id * stride
            stem = f"{info.sequence_id}_cam0_W{window_id:05d}_F{start_frame:06d}"
            full_path = full_dir / f"{stem}.pt"
            roi_path = roi_dir / f"{stem}.pt"

            torch.save(stack_uint8_clip(full_rgb), full_path)
            torch.save(stack_uint8_clip(roi_rgb), roi_path)

            rows.append({
                "clip_path": str(full_path.relative_to(output_root)),
                "roi_path": str(roi_path.relative_to(output_root)),
                "label": str(info.label),
                "dataset": "UR-Fall",
                "split": split_name,
                "sequence_id": info.sequence_id,
                "kind": info.kind,
                "sequence_number": str(info.number),
                "camera": "0",
                "window_id": str(window_id),
                "start_frame": str(start_frame),
            })
            window_id += 1

            for _ in range(min(stride, len(frames_q))):
                frames_q.popleft()
                dets_q.popleft()

    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = [
        "clip_path", "roi_path", "label", "dataset", "split",
        "sequence_id", "kind", "sequence_number", "camera",
        "window_id", "start_frame",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_preprocess(args) -> Dict[str, int]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    infos = discover_sources(data_root)
    if not infos:
        raise RuntimeError(
            "No UR-Fall camera-0 RGB sequences discovered. Expected official-style "
            "names such as fall-01-cam0-rgb.zip / adl-01-cam0-rgb.zip "
            "or extracted directories with the same stems."
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

    id_to_split = {}
    for name in ("train", "val", "test"):
        for seq_id in split[name]:
            id_to_split[seq_id] = name

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
    print(f"[UR-Fall] sequences discovered: {len(infos)}")

    for idx, info in enumerate(infos, start=1):
        split_name = id_to_split[info.sequence_id]
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
                "dataset": "UR-Fall",
                "camera": 0,
                "clip_len": args.clip_len,
                "stride": args.stride,
                "image_size": args.image_size,
                "split": split,
                "clip_counts": counts,
                "expected_paper_total_clips": 1648,
                "roi_enabled": not args.skip_roi,
                "yolo_conf": args.yolo_conf,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[UR-Fall] clip counts:", counts)
    if counts["total"] != 1648:
        print(
            "[NOTE] The paper reports 1,648 UR-Fall clips. Your generated total "
            f"is {counts['total']}. Check the downloaded RGB archives/frames. "
            "Do not force the count by duplicating or deleting clips."
        )
    return counts


def make_synthetic_urfall(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # Create all official sequence stems with 20 tiny JPEG frames in zip archives.
    items = [("fall", i) for i in range(1, 31)] + [("adl", i) for i in range(1, 41)]
    for kind, num in items:
        zpath = root / f"{kind}-{num:02d}-cam0-rgb.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for i in range(20):
                img = np.full((32, 48, 3), (num * 7 + i) % 255, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", img)
                assert ok
                zf.writestr(f"frame_{i:04d}.jpg", encoded.tobytes())
    return root


def smoke_test() -> None:
    print("[Smoke test] UR-Fall preprocessing")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = make_synthetic_urfall(root / "raw")
        out = root / "out"

        class A:
            data_root = str(data)
            output_root = str(out)
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
        assert counts["total"] == 70
        assert counts["train"] == 49
        assert counts["val"] == 7
        assert counts["test"] == 14
        for name in ("train.csv", "val.csv", "test.csv", "split_used.json"):
            assert (out / name).exists(), name
        print("[Smoke test] PASS")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess UR-Fall for RMTD-FD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--data-root", type=str)
    p.add_argument("--output-root", type=str)
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
