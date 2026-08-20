## Installation

Clone this repository:

```bash
git clone https://github.com/dafei7225-ctrl/RMTD-FD.git
cd RMTD-FD
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Before running the full pipeline, the main scripts can be checked using their built-in smoke tests:

```bash
python data/preprocess_upfall_rmtd_fd.py --smoke-test
python data/preprocess_urfall_rmtd_fd.py --smoke-test
python distillation/pretrain_rmtd_fd_teachers.py --smoke-test
python distillation/train_rmtd_fd_distill.py --smoke-test
python evaluation/evaluate_rmtd_fd.py --smoke-test
python inference/inference_online_alarm_rmtd_fd.py --smoke-test
```

---

## Dataset Preparation

RMTD-FD is evaluated on the **UP-Fall** and **UR-Fall** datasets.

The original datasets are not redistributed in this repository. Please download them from their official sources before preprocessing.

### UP-Fall

Dataset website:

https://sites.google.com/up.edu.mx/har-up/

The preprocessing script generates:

* 16-frame RGB clips;
* stride of 8 frames;
* 224 × 224 inputs;
* binary fall / non-fall labels;
* full-frame clips for T1, T2, and the student;
* human-ROI clips for T3;
* `train.csv`, `val.csv`, and `test.csv`.

Run:

```bash
python data/preprocess_upfall_rmtd_fd.py \
    --data-root /path/to/UP-Fall \
    --output-root /path/to/processed/UP_Fall \
    --camera 1 \
    --split-json /path/to/upfall_split.json
```

If the exact experimental split file is unavailable, the script can generate a deterministic 12/2/3 cross-subject split:

```bash
python data/preprocess_upfall_rmtd_fd.py \
    --data-root /path/to/UP-Fall \
    --output-root /path/to/processed/UP_Fall \
    --camera 1 \
    --split-seed 42
```

> For exact reproduction of the reported results, the original experimental subject split should be used whenever available.

### UR-Fall

Dataset website:

https://fenix.ur.edu.pl/~mkepski/ds/uf.html

Run:

```bash
python data/preprocess_urfall_rmtd_fd.py \
    --data-root /path/to/UR-Fall \
    --output-root /path/to/processed/UR_Fall \
    --split-json /path/to/urfall_split.json
```

If the exact experimental split file is unavailable, a deterministic stratified 49/7/14 sequence-level split can be generated using:

```bash
python data/preprocess_urfall_rmtd_fd.py \
    --data-root /path/to/UR-Fall \
    --output-root /path/to/processed/UR_Fall \
    --split-seed 42
```

The default stratified split preserves approximately consistent fall/non-fall proportions across training, validation, and testing subsets.

---

## Teacher Pretraining

Three role-specific teachers are independently pretrained:

| Teacher                                 | Backbone        | Input                |
| --------------------------------------- | --------------- | -------------------- |
| T1: Temporal Context Teacher            | UniFormer-B     | Full-frame RGB clips |
| T2: Local Action Discrimination Teacher | UniFormer-S     | Full-frame RGB clips |
| T3: Human-Centered Teacher              | ROI-UniFormer-B | Human ROI clips      |

The paper settings are used as defaults:

* 16 frames;
* 224 × 224 input resolution;
* AdamW optimizer;
* learning rate = `1e-4`;
* weight decay = `0.05`;
* batch size = `16`;
* 50 epochs.

### T1 — Temporal Context Teacher

```bash
python distillation/pretrain_rmtd_fd_teachers.py \
    --teacher t1 \
    --train-manifest /path/to/processed/DATASET/train.csv \
    --val-manifest /path/to/processed/DATASET/val.csv \
    --model-factory <T1_MODEL_FACTORY> \
    --output-dir runs/t1 \
    --amp
```

### T2 — Local Action Discrimination Teacher

```bash
python distillation/pretrain_rmtd_fd_teachers.py \
    --teacher t2 \
    --train-manifest /path/to/processed/DATASET/train.csv \
    --val-manifest /path/to/processed/DATASET/val.csv \
    --model-factory <T2_MODEL_FACTORY> \
    --output-dir runs/t2 \
    --amp
```

### T3 — Human-Centered Teacher

```bash
python distillation/pretrain_rmtd_fd_teachers.py \
    --teacher t3 \
    --train-manifest /path/to/processed/DATASET/train.csv \
    --val-manifest /path/to/processed/DATASET/val.csv \
    --model-factory <T3_MODEL_FACTORY> \
    --output-dir runs/t3 \
    --amp
```

`<T1_MODEL_FACTORY>`, `<T2_MODEL_FACTORY>`, and `<T3_MODEL_FACTORY>` must point to the corresponding model constructors in `module:function` or `file.py:function` format.

---

## RMTD-FD Distillation Training

After pretraining the three teachers, the RMTD-FD student is trained using role-aware multi-teacher distillation.

During distillation:

* all teacher parameters are frozen;
* only the UniFormer-XS student is optimized;
* teacher reliability and sample role demand jointly determine sample-level teacher weights;
* the paper hyperparameters are used as defaults.

Run:

```bash
python distillation/train_rmtd_fd_distill.py \
    --train-manifest /path/to/processed/DATASET/train.csv \
    --val-manifest /path/to/processed/DATASET/val.csv \
    --student-factory <STUDENT_MODEL_FACTORY> \
    --t1-factory <T1_MODEL_FACTORY> \
    --t2-factory <T2_MODEL_FACTORY> \
    --t3-factory <T3_MODEL_FACTORY> \
    --t1-ckpt /path/to/t1_checkpoint.pt \
    --t2-ckpt /path/to/t2_checkpoint.pt \
    --t3-ckpt /path/to/t3_checkpoint.pt \
    --temporal-blocks <K> \
    --output-dir runs/rmtd_fd_distill \
    --amp
```

Main default settings:

| Parameter                          |     Value |
| ---------------------------------- | --------: |
| Frames                             |        16 |
| Input size                         | 224 × 224 |
| Batch size                         |         8 |
| Epochs                             |       100 |
| Learning rate                      |      1e-4 |
| Weight decay                       |      0.05 |
| Distillation temperature τ         |         4 |
| Distillation coefficient λ         |       0.5 |
| Attention coefficient λatt         |       1.0 |
| Reliability temperature τr         |       1.0 |
| Demand coefficient γ               |       0.5 |
| Weight-allocation temperature τα   |       1.0 |
| Teacher calibration temperature Tc |       2.0 |

`<K>` denotes the number of temporal blocks used for temporal role-demand estimation and must match the setting used in the reported experiments.

---

## Evaluation

The final RMTD-FD model is evaluated using the **student model only**.

The evaluation script reports:

* Accuracy;
* Precision;
* Recall;
* F1-score.

The fall class is treated as the positive class.

### UP-Fall

```bash
python evaluation/evaluate_rmtd_fd.py \
    --manifest /path/to/processed/UP_Fall/test.csv \
    --model-factory <STUDENT_MODEL_FACTORY> \
    --checkpoint /path/to/best_student.pt \
    --output-dir runs/eval_upfall \
    --online-eval \
    --amp
```

### UR-Fall

```bash
python evaluation/evaluate_rmtd_fd.py \
    --manifest /path/to/processed/UR_Fall/test.csv \
    --model-factory <STUDENT_MODEL_FACTORY> \
    --checkpoint /path/to/best_student.pt \
    --output-dir runs/eval_urfall \
    --online-eval \
    --amp
```

The evaluation output contains:

```text
evaluation_summary.json
predictions.csv
online_alarms.csv
sequence_summary.csv
```

The default online decision parameters are:

* moving-average window: `M = 3`;
* persistence: `C = 2`;
* alarm threshold: `θ = 0.65`.

---

## Online Inference and Fall Alarm

Only the trained UniFormer-XS student model is required during online inference.

No teacher network, ROI teacher branch, or dynamic routing module is used.

### Video File

```bash
python inference/inference_online_alarm_rmtd_fd.py \
    --source /path/to/input_video.mp4 \
    --model-factory <STUDENT_MODEL_FACTORY> \
    --checkpoint /path/to/best_student.pt \
    --output-dir runs/online_demo \
    --display \
    --save-video \
    --amp
```

### Webcam

```bash
python inference/inference_online_alarm_rmtd_fd.py \
    --source 0 \
    --model-factory <STUDENT_MODEL_FACTORY> \
    --checkpoint /path/to/best_student.pt \
    --output-dir runs/webcam_demo \
    --display \
    --amp
```

Default online inference settings:

| Parameter               |     Value |
| ----------------------- | --------: |
| Clip length             | 16 frames |
| Stride                  |  8 frames |
| Input resolution        | 224 × 224 |
| Moving-average window M |         3 |
| Consecutive windows C   |         2 |
| Alarm threshold θ       |      0.65 |

The inference script produces:

```text
window_predictions.csv
alarm_events.csv
inference_summary.json
annotated_output.mp4
```

The annotated video is generated when `--save-video` is enabled.
