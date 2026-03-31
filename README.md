# User Engagement-Aware Gesture Recognition for Human-Robot Interaction in Multi-Person Settings

> **Published at RAAI 2025, Singapore**  
> Snehal Walunj, Pranav Walunj, Khalil Abubaid, Tatjana Legler, Martin Ruskowski  
> German Research Center for Artificial Intelligence (DFKI) · University of Kaiserslautern-Landau (RPTU), Germany

---

## Overview

In shared industrial workspaces, robots must reliably identify *which* person they should respond to and not just *what* gesture is being made. This paper introduces a **two-step hierarchical detection framework** that solves this fundamental user-identification problem, enabling truly user-aware human-robot collaboration.

The system first identifies the primary worker and their engagement state using body skeleton data, then and only then it activates gesture recognition via hand landmarks. By decoupling **WHO** from **WHAT**, the framework eliminates false triggers from bystanders and creates a safer, more reliable interaction loop.

---

## The Problem

In real industrial settings, multiple people are always present around robot workstations. Existing gesture recognition systems treat all humans equally and cannot identify who the robot should actually respond to. This leads to:

- **Safety Risks** : Unexpected robot actions triggered by unintended users
- **Wrong Actions** : Robot responding to the wrong person
- **False Triggers** : Bystanders accidentally activating commands
- **Workflow Disruption** : Constant interruptions from random gestures

---

## Solution: Two-Step Hierarchical Framework

```
ZED-2i Camera
     │
     ▼
┌─────────────────────────────────┐
│  STEP 1: User ID & Engagement   │  ← Random Forest on 18-keypoint skeleton data
│  Identifies main worker         │
│  Checks engagement state        │
└────────────┬────────────────────┘
             │  Only if: Main Worker PRESENT + ENGAGED (3s stability)
             ▼
┌─────────────────────────────────┐
│  STEP 2: Gesture Recognition    │  ← MLP on MediaPipe hand landmarks (21 keypoints)
│  Recognizes functional command  │
└────────────┬────────────────────┘
             │
             ▼
      Robot Action
   (Give Part / Stop / Continue)
```

**Key insight:** Stage 2 is completely inactive unless Stage 1 confirms the main worker is present and engaged for 3 consecutive seconds : physically excluding all bystanders from detection.

---

## System Architecture

### Stage 1 : Engagement Detection

| Component | Details |
|-----------|---------|
| Sensor | ZED-2i stereo camera |
| Input | 18 body 3D keypoints (skeleton) over 3-frame temporal window |
| Classifier | Random Forest |
| Output | `Engaged` / `Disengaged` / `Not Present` |
| Stability Window | 3 consecutive seconds of `Engaged` predictions before Stage 2 activates |

**Engagement State Labels:**

| Label | Description |
|-------|-------------|
| `Engaged` | Main worker present and actively attentive to the robot |
| `Disengaged` | Main worker present but not focused on the robot |
| `Not Present` | Only side workers / bystanders in the scene |

### Stage 2 : Gesture Recognition

| Component | Details |
|-----------|---------|
| Input | Cropped ROI (bounding box around engaged worker) |
| Hand Tracking | MediaPipe Hands : 21 normalized hand landmarks |
| Classifier | Multi-Layer Perceptron (MLP) |
| Output | Gesture command |

**Gesture Labels:**

| Gesture | Description |
|---------|-------------|
| `Give` | Request part handover |
| `Stop` | Halt current operation |
| `Continue` | Resume / proceed workflow |
| `Unknown` | Unrecognized hand pose |

**Dynamic ROI Extraction:** Once Stage 1 confirms the engaged worker, a bounding box is computed from their skeleton keypoints. Only this cropped region is passed to Stage 2 : bystanders are physically outside the image, making false positives architecturally impossible.

---

## Dataset

**Participants:** 3 participants · Ages 25–35 · Heights 150–182 cm  
**Recording:** 8 sessions · ~2 hours total

### 5 Collection Scenarios

| # | Scenario |
|---|----------|
| 1 | Engaged main worker alone |
| 2 | Disengaged main worker alone |
| 3 | No main worker, only side workers |
| 4 | Engaged main worker + side workers |
| 5 | Disengaged main worker + side workers |

### Dataset Statistics

| Stage | Samples | Source |
|-------|---------|--------|
| Stage 1 (Engagement) | 10,100 labelled instances · 6,235 images | All 5 scenarios |
| Stage 2 (Gestures) | 4,538 gesture samples | Scenarios 1 & 4 only |

---

## Results

### Stage 1 : Engagement Classification

| Classifier | Accuracy |
|-----------|----------|
| Logistic Regression | 88.5% |
| Multi-Layer Perceptron | 92.1% |
| **Random Forest**  | **95.4%** |

**Random Forest Full Metrics:** Precision: 0.96 · Recall: 0.94 · F1-Score: 0.95

### Stage 2 : Gesture Recognition

| Classifier | Accuracy |
|-----------|----------|
| Logistic Regression | 91.2% |
| Random Forest | 94.5% |
| **Multi-Layer Perceptron**  | **98.1%** |

**MLP Full Metrics:** Precision: 0.99 · Recall: 0.97 · F1-Score: 0.98

### End-to-End Pipeline Performance

Evaluated on **60 balanced test instances** across all 3 engagement states × 4 gesture types × 5 samples each.

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **83.33%** |
| Correctly Handled | 50 / 60 |
| True Positives (full pipeline success) | 18 |
| True Negatives (system correctly stays inactive) | 32 |
| Precision & Recall | 0.78 (balanced) |

**Accuracy formula:**  
`A_total = (C_pass + C_stop) / N × 100% = (18 + 32) / 60 × 100% = 83.33%`

The end-to-end accuracy is lower than individual stage accuracies because both stages must succeed sequentially, this reflects real-world performance in complex multi-person scenarios that single-stage systems cannot handle at all.

---

## Why Two Steps?

| Traditional Approach | Our Two-Step Approach |
|---------------------|----------------------|
| No user identification | User identified first (Stage 1) |
| Every hand gesture may trigger action | Stage 2 only runs when worker is `Engaged` |
| High false trigger rate from bystanders | Zero false positives,i.e. bystanders physically excluded |
| Safety risks from unintended gestures | Robot responds only to the intended operator |

---

## Key Achievements

- **Solved the user-identification problem** : First system to reliably identify the engaged worker in multi-person industrial settings
- **Eliminated false positives** : Two-step architecture prevents any bystander gesture from triggering the robot
- **Strong cascaded performance** : Stage 1: 95.4% · Stage 2: 98.1% · End-to-end: 83.3%
- **Safer HRI** : Robot only responds to the intended operator at the right time


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{walunj2025hri,
  title     = {User Engagement-Aware Gesture Recognition for Human-Robot Interaction System in Multi-Person Settings},
  author    = {Walunj, Snehal and Walunj, Pranav and Abubaid, Khalil and Legler, Tatjana and Ruskowski, Martin},
  booktitle = {Proceedings of the International Conference on Robotics, Automation and Artificial Intelligence (RAAI)},
  year      = {2025},
  address   = {Singapore}
}
```

---

## Acknowledgements

Funded by the **WSKL Chair, RPTU Kaiserslautern-Landau**.  
Special thanks to the RPTU Kaiserslautern colleagues, DFKI research team, all study participants, and the conference reviewers.
