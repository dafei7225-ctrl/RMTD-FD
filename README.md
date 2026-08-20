# RMTD-FD

Official implementation of **RMTD-FD: Role-Aware Multi-Teacher Distillation for Lightweight Video-Based Fall Detection**.

RMTD-FD is a role-aware multi-teacher knowledge distillation framework designed for lightweight video-based fall detection. It transfers complementary temporal, boundary-discrimination, and human-centered knowledge from multiple role-specific teachers to a lightweight student model through a **Reliability–Demand Factorized Dynamic Routing** mechanism.

During inference, all teacher models and the dynamic routing module are removed, and only the lightweight **UniFormer-XS** student model is retained.

> **Repository status:** The source code and reproduction instructions are currently being organized and will be updated progressively.

---

## Overview

Lightweight video-based fall detection faces several challenges, including:

* fall-stage evidence dispersed across video frames;
* confusion between actual falls and fall-like actions such as rapid sitting, squatting, and bending;
* interference caused by occlusion, complex backgrounds, and multiple moving persons.

To address these issues, RMTD-FD formulates multi-teacher distillation as a **sample-conditioned role knowledge routing problem**.

The proposed framework contains three role-specific teachers:

| Teacher                             | Backbone        | Role-specific Knowledge                     | Distillation Signal              |
| ----------------------------------- | --------------- | ------------------------------------------- | -------------------------------- |
| Temporal Context Teacher            | UniFormer-B     | Temporal dynamics and fall-stage continuity | Soft labels + temporal attention |
| Local Action Discrimination Teacher | UniFormer-S     | Fine-grained local action discrimination    | Shallow local features           |
| Human-Centered Teacher              | ROI-UniFormer-B | Human-centered spatial representation       | Human-centered features          |

The student model is:

* **Student:** UniFormer-XS
* **Input:** full-frame RGB video clips
* **Inference:** student-only deployment

---

## Method

### Role-Specific Multi-Teacher Distillation

RMTD-FD transfers three complementary types of knowledge to the lightweight student model.

#### 1. Temporal Context Teacher

The temporal context teacher uses **UniFormer-B** and provides:

* soft-label supervision;
* temporal attention supervision;
* knowledge related to cross-frame temporal dependencies and fall-stage continuity.

Temporal attention is extracted from the last SplitSABlock of Stage 4.

#### 2. Local Action Discrimination Teacher

The local action discrimination teacher uses **UniFormer-S**.

Shallow local spatiotemporal features from **Stage 1 and Stage 2** are transferred to the student to improve the discrimination between falls and visually similar actions such as rapid sitting, bending, and squatting.

#### 3. Human-Centered Teacher

The human-centered teacher uses **ROI-UniFormer-B** and takes human-region ROI clips as input.

Human ROIs are generated using a **COCO-pretrained YOLOv8n** detector. Only the person class is retained.

The full-frame UniFormer-XS student does not require ROI extraction during inference.

---

## Reliability–Demand Dynamic Teacher Weighting

Instead of assigning fixed or uniform weights to multiple teachers, RMTD-FD dynamically determines the contribution of each teacher for every input sample.

The routing mechanism jointly considers:

### Teacher Reliability

Teacher reliability evaluates whether the supervision provided by a teacher is reliable for the current sample.

### Sample Role Demand

Three role-demand terms are used:

* **Temporal role demand:** reflects instability across temporal segments;
* **Boundary-discrimination role demand:** reflects classification uncertainty;
* **Human-centered role demand:** reflects the discrepancy between human-centered and full-frame predictions.

Teacher reliability and role demand are jointly combined to produce sample-level dynamic teacher weights.

The final multi-teacher distillation loss is obtained through weighted aggregation of the three role-specific teacher losses.

The dynamic routing mechanism is used **only during training**.

---

## Framework

The overall training framework of RMTD-FD consists of:

```text
Continuous RGB Video
        │
        ├── Temporal Context Teacher (UniFormer-B)
        │       └── Temporal attention + soft labels
        │
        ├── Local Action Discrimination Teacher (UniFormer-S)
        │       └── Shallow local features
        │
        ├── Human ROI Extraction
        │       └── ROI-UniFormer-B
        │               └── Human-centered features
        │
        └── Student Model (UniFormer-XS)
                    │
                    ▼
       Reliability–Demand Dynamic Routing
                    │
                    ▼
         Weighted Multi-Teacher Distillation
                    │
                    ▼
             Trained Student Model
```

The framework figures used in the paper can be placed in an `assets/` directory, for example:

```text
assets/
├── training_framework.png
├── inference_pipeline.png
└── dynamic_teacher_weighting.png
```

---

## Online Inference

RMTD-FD performs video-based fall detection using a sliding-window strategy.

The input video is divided into clips using:

| Parameter        |   Setting |
| ---------------- | --------: |
| Clip length      | 16 frames |
| Sliding stride   |  8 frames |
| Input resolution | 224 × 224 |

The student model produces a fall probability for each sliding window.

For online fall decision:

1. the window-level fall probabilities are smoothed using a causal moving average;
2. the smoothed probability is compared with a predefined threshold;
3. the threshold condition must persist for multiple consecutive windows;
4. consecutive positive windows are merged into a single fall alarm event.

The parameters used in the experiments are:

| Parameter                 | Value |
| ------------------------- | ----: |
| Moving-average window (M) |     3 |
| Persistence windows (C)   |     2 |
| Alarm threshold (\theta)  |  0.65 |

Only the **UniFormer-XS student model** is retained during inference.

---

## Datasets

Experiments are conducted on two publicly available video-based fall detection datasets:

* **UP-Fall**
* **UR-Fall**

The original datasets are not redistributed in this repository. Please download them from their official sources.

### UP-Fall Detection Dataset

Official website:

https://sites.google.com/up.edu.mx/har-up/

Reference:

> L. Martínez-Villaseñor, H. Ponce, J. Brieva, E. Moya-Albor, J. Núñez-Martínez, and C. Peñafort-Asturiano,
> “UP-Fall Detection Dataset: A Multimodal Approach,”
> *Sensors*, vol. 19, no. 9, 1988, 2019.
> DOI: 10.3390/s19091988

Experimental protocol used in RMTD-FD:

| Item                      | Setting             |
| ------------------------- | ------------------- |
| Subjects                  | 17                  |
| Original classes          | 11 action classes   |
| Classes used              | Fall / Non-fall     |
| Video resolution          | 640 × 480           |
| Split protocol            | Fixed cross-subject |
| Train / Validation / Test | 12 / 2 / 3 subjects |
| Generated clips           | 17,901              |

### UR-Fall Detection Dataset

Official website:

https://fenix.ur.edu.pl/~mkepski/ds/uf.html

Reference:

> B. Kwolek and M. Kepski,
> “Human fall detection on embedded platform using depth maps and wireless accelerometer,”
> *Computer Methods and Programs in Biomedicine*, vol. 117, no. 3, pp. 489–501, 2014.
> DOI: 10.1016/j.cmpb.2014.09.005

Experimental protocol used in RMTD-FD:

| Item                      | Setting                         |
| ------------------------- | ------------------------------- |
| Original sequences        | 70                              |
| Fall sequences            | 30                              |
| ADL sequences             | 40                              |
| Classes used              | Fall / Non-fall                 |
| Video resolution          | 640 × 480                       |
| Split protocol            | Fixed stratified sequence-level |
| Train / Validation / Test | 49 / 7 / 14 sequences           |
| Generated clips           | 1,648                           |

For both datasets, subject-level or original-sequence-level splitting is performed **before** sliding-window clip generation to prevent overlapping clips from the same subject or original sequence from appearing in different subsets.

---

## Dataset Preparation

After downloading the original datasets, they can be organized under:

```text
datasets/
├── UP-Fall/
│   └── ...
└── UR-Fall/
    └── ...
```

Detailed preprocessing and annotation conversion scripts will be provided with the source code.

---

## Implementation Details

The main experimental settings reported in the paper are:

| Configuration                               | Setting          |
| ------------------------------------------- | ---------------- |
| Deep learning framework                     | PyTorch          |
| GPU                                         | NVIDIA RTX 4090  |
| Input frames                                | 16               |
| Input resolution                            | 224 × 224        |
| Student                                     | UniFormer-XS     |
| Temporal Context Teacher                    | UniFormer-B      |
| Local Action Discrimination Teacher         | UniFormer-S      |
| Human-Centered Teacher                      | ROI-UniFormer-B  |
| Optimizer                                   | AdamW            |
| Teacher learning rate                       | (1\times10^{-4}) |
| Distillation learning rate                  | (1\times10^{-4}) |
| Weight decay                                | 0.05             |
| Teacher batch size                          | 16               |
| Distillation batch size                     | 8                |
| Teacher training epochs                     | 50               |
| Distillation epochs                         | 100              |
| Distillation temperature (\tau)             | 4                |
| Balance coefficient (\lambda)               | 0.5              |
| Attention weight (\lambda_{att})            | 1.0              |
| Reliability temperature (\tau_r)            | 1.0              |
| Demand weight (\gamma)                      | 0.5              |
| Weight-allocation temperature (\tau_\alpha) | 1.0              |
| Teacher calibration temperature (T_c)       | 2.0              |

### Human ROI Extraction

Human ROIs used by the human-centered teacher are generated with a **COCO-pretrained YOLOv8n** model from Ultralytics.

Settings:

* detection class: `person`;
* confidence threshold: `0.25`;
* single-person scenes: select the highest-confidence human bounding box;
* multi-person scenes: select the subject appearing most consistently across frames with the largest average bounding-box area;
* detection failure: use the original full-frame input as fallback;
* ROI images are resized to the model input resolution.

ROI extraction is required only for training the human-centered teacher and is **not required during student inference**.

---

## Environment

The implementation is based on **PyTorch**.

Exact software versions will be added together with the final released environment file.

```text
Python      : TBD
PyTorch     : TBD
CUDA        : TBD
torchvision : TBD
```

Dependencies will be installable using:

```bash
pip install -r requirements.txt
```

---

## Training

RMTD-FD training consists of two main stages.

### Stage 1: Teacher Training

The three role-specific teacher models are trained independently on the same training set:

* UniFormer-B temporal context teacher;
* UniFormer-S local action discrimination teacher;
* ROI-UniFormer-B human-centered teacher.

### Stage 2: Multi-Teacher Distillation

During joint distillation:

* all teacher parameters are frozen;
* the UniFormer-XS student is optimized;
* role-specific distillation losses are dynamically weighted using the Reliability–Demand routing mechanism.

Exact training commands will be provided together with the released source code.

```bash
# Example only — final command will be updated
python train.py [arguments]
```

---

## Evaluation

The fall class is treated as the positive class.

The following clip-level metrics are reported:

* Accuracy
* Precision
* Recall
* F1-score

Exact evaluation commands will be provided with the evaluation scripts.

```bash
# Example only — final command will be updated
python test.py [arguments]
```

---

## Main Results

### UP-Fall

| Method                  | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ----------------------- | -----------: | ------------: | ---------: | -----------: |
| Baseline (UniFormer-XS) |        95.25 |         90.33 |      90.90 |        90.61 |
| **RMTD-FD**             |    **97.51** |     **92.94** |  **92.65** |    **92.80** |

### UR-Fall

| Method                  | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ----------------------- | -----------: | ------------: | ---------: | -----------: |
| Baseline (UniFormer-XS) |        91.32 |         90.70 |      89.53 |        90.11 |
| **RMTD-FD**             |    **92.41** |     **90.85** |  **91.83** |    **91.33** |

---

## Model Complexity and Inference Efficiency

During inference, RMTD-FD removes all teacher networks and the dynamic routing module.

Only the UniFormer-XS student model is retained.

| Model                      | Parameters |     FLOPs |    FPS |
| -------------------------- | ---------: | --------: | -----: |
| UniFormer-B                |     50.8 M |     146 G |     18 |
| **RMTD-FD (Student only)** | **12.8 M** | **3.4 G** | **30** |

The reported inference speed is measured using an **NVIDIA RTX 4090**.

---

## Ablation Results

### Role-Specific Teachers

The three teachers provide complementary knowledge.

On the UP-Fall test subsets:

* the Local Action Discrimination Teacher reduces the false-alarm rate on fall-like actions;
* the Human-Centered Teacher improves performance under complex-background conditions;
* combining all teachers provides further performance improvement;
* Reliability–Demand Dynamic Weighting further improves the overall result.

| Configuration    | UP-Fall F1 (%) | UR-Fall F1 (%) | Fall-Like Action FAR (%) | Complex Background F1 (%) |
| ---------------- | -------------: | -------------: | -----------------------: | ------------------------: |
| Baseline         |          90.61 |          90.11 |                     13.2 |                     90.26 |
| + T1             |          91.31 |          90.48 |                     12.4 |                     90.73 |
| + T2             |          91.28 |          90.54 |                      9.3 |                     90.58 |
| + T3             |          91.35 |          90.13 |                     11.7 |                     90.87 |
| + T1 + T2 + T3   |          92.21 |          90.86 |                      9.1 |                     91.72 |
| **Full RMTD-FD** |      **92.80** |      **91.33** |                  **7.8** |                 **92.53** |

### Dynamic Weighting

| Weighting Strategy            | F1-score (%) | Accuracy (%) |
| ----------------------------- | -----------: | -----------: |
| Baseline                      |        90.61 |        95.25 |
| Equal Weight                  |        90.87 |        95.30 |
| Confidence-only               |        91.32 |        95.92 |
| Reliability-only              |        91.16 |        95.74 |
| **Reliability + Role Demand** |    **92.80** |    **97.51** |

---

## Visualization

The paper provides two types of visualization analyses:

* **sample-conditioned dynamic teacher weights** for slow-fall, rapid-sitting, and complex-background samples;
* **Grad-CAM comparisons** between the baseline and RMTD-FD under complex-background and rapid-sitting scenarios.

Corresponding visualization scripts and figures will be added to this repository.

---

## Repository Structure

The final repository is expected to follow a structure similar to:

```text
RMTD-FD/
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
├── datasets/
├── models/
├── utils/
│
├── train.py
├── test.py
├── inference.py
│
└── assets/
    ├── training_framework.png
    ├── inference_pipeline.png
    └── dynamic_teacher_weighting.png
```

The directory structure will be updated according to the final released implementation.

---

## Pretrained Models

Pretrained teacher and student checkpoints will be added if redistribution and repository file-size constraints permit.

Model links and corresponding configurations will be provided here after release.

---

## Citation

If you find RMTD-FD useful for your research, please consider citing our work.

The final BibTeX entry will be updated after publication.

```bibtex
@article{RMTDFD,
  title  = {RMTD-FD: Role-Aware Multi-Teacher Distillation for Lightweight Video-Based Fall Detection},
  author = {Yan Zhu and Pengfei Cai and Bomin Liu and Rui Zhou},
  journal = {To be updated},
  year = {To be updated}
}
```

---

## Authors

* Yan Zhu
* Pengfei Cai
* Bomin Liu
* Rui Zhou

For correspondence:

* Rui Zhou: `ruizhou9920@gmail.com`
* Bomin Liu: `liubm@sdju.edu.cn`

---

## Code Availability

The source code of RMTD-FD is available in this repository:

https://github.com/dafei7225-ctrl/RMTD-FD

---

## License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the authors of the **UP-Fall** and **UR-Fall** datasets for making their datasets publicly available to the research community.
