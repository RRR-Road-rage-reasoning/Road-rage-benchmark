# Road-Rage-Benchmark

**Road-Rage-Benchmark** targets *pre-response road rage regulation* by reasoning about situational triggers **before** aggressive driving behaviors occur.  
We introduce **Road Rage Reasoning** as a new task, release **RoadRageBench** with rich real-world annotations, conduct extensive evaluations of open-source Vision-Language Models (VLMs), and propose a **real-time VLM-based reasoning framework** to support proactive in-vehicle safety systems.

---

## 🎯 Task Definition & Dataset

### Road Rage Reasoning Tasks

Inspired by the human emotional response chain (**Situation → Attention → Appraisal → Response**), we define **Road Rage Reasoning (R³)** as proactively judging hazardous driving scenarios **before** aggressive reactions occur.  
We design a **three-level progressive task framework** with dashcam video frames sampled at **2 FPS** as unified visual input:

1. **Frame-level Environmental Cue Perception**  
   Extract fine-grained environmental and traffic cues from individual frames, including:
   - Physical environment: weather, time of day, road type, road condition  
   - Traffic conditions: lane number, ego lane position/state, lane markings  
   - Key surrounding objects: relative position, distance, and behaviors (e.g., lane cutting, sudden braking, jaywalking)

2. **Video-level Event Recognition**  
   Identify whether the following **9 interaction events** occur in a video:
   1) Unsafe lane change  
   2) Lane change without signaling  
   3) Illegal lane change across solid lines  
   4) Unsafe overtaking  
   5) Sudden braking at close distance  
   6) Pedestrians or non-motorized users crossing  
   7) Repeated braking to obstruct traffic  
   8) Repeated lane cutting  
   9) Traffic congestion

3. **Video-level Scene Classification**  
   Each video is classified into three **independent binary categories**:
   - **Dangerous**: direct safety threats (events 1–6)  
   - **Aggressive**: intentional disruptive behaviors (events 7–8)  
   - **Obstructive**: traffic efficiency reduction (event 9)

---

### RoadRageBench Dataset

We construct **RoadRageBench**, the first real-world dataset dedicated to road rage reasoning, using a **three-step pipeline**:

1. **Data Collection & Preliminary Annotation**  
   Dashcam videos are collected from YouTube and Bilibili using targeted keywords (e.g., *road rage*, *unsafe driving*). Videos are filtered to ensure first-person view, clear visuals, consistent traffic rules, and clear road rage scenarios. Initial event and scene labels are assigned.

2. **Personnel Review & Secondary Screening**  
   Three reviewers with driving experience score anger intensity and scene clarity. Videos with low consensus or insufficient samples are removed, resulting in **81 road rage videos** covering all 9 events.

3. **Frame-level Annotation**  
   Videos are uniformly downsampled to **2 FPS**. Each frame is annotated with detailed environmental and traffic cues to support fine-grained perception tasks.

To balance the dataset, **19 non-rage (safe driving) videos** are added with video-level labels only.

**Final Dataset Statistics**:
- **100 videos** (81 road rage + 19 non-rage)  
- **2,299 frames**  
- **22,000+ fine-grained annotations**

RoadRageBench fully supports the reasoning chain:  
**Environmental Cue Perception → Event Recognition → Scene Classification**, enabling systematic evaluation of VLMs for proactive in-car safety.

## 🧠VLM-based reasoning framework

### Visual Feature Extraction

Each input video is represented as a sequence of RGB frames.  
We uniformly sample **20 frames per video** to balance temporal coverage and efficiency.

The sampled frames are fed into the **visual encoder of a pretrained Vision-Language Model (VLM)**.  
Hidden states from the final vision layer are extracted as visual features.

To obtain a compact video representation, we apply a two-stage average pooling strategy:

1. **Token-level pooling**: aggregates spatial information within each frame  
2. **Temporal pooling**: aggregates information across frames  

This produces a single **video-level feature vector**, which captures the overall scene semantics and serves as input to the classifier.

---

### Scene Discrimination

A lightweight **Multilayer Perceptron (MLP)** is used for scene classification:

- Input: video-level visual feature
- Architecture:
  - FC → 128 → BatchNorm → ReLU → Dropout (0.3)
  - FC → 64 → BatchNorm → ReLU → Dropout (0.3)
  - FC → 3 (Dangerous / Aggressive / Obstructive)

This design balances classification performance and real-time efficiency.

---

### Framework overview

<img width="2475" height="1120" alt="fig3-framework_01" src="https://github.com/user-attachments/assets/73fa6b98-7ab6-41be-ae7e-b186a324dd09" />

---

## ⚙️ Environment Setup

### Requirements

- Python >= 3.9+
- PyTorch >= 2.x (with CUDA support recommended)
- Tested with CUDA 12.x

### Installation

```bash
cd requirements
pip install -r x_requirements.txt
```

---

### Model Weights

This project relies on several pretrained Vision-Language Models (VLMs) as visual
feature extractors. Due to their large size and licensing constraints, the model
weights are **not included** in this repository.

Please download the corresponding pretrained models from their official sources:

- **InternVL3-8B**: available from the official InternVL repository or Hugging Face Hub  
- **LLaVA-onevision-qwen2-7B / LLaVA-NeXT-Video-7B**: available from the official LLaVA repository or Hugging Face Hub  
- **Qwen2.5-VL-7B-Instruct / Qwen2.5-VL-32B-Instruct**: available from the official Qwen repository or Hugging Face Hub  

After downloading, place the model checkpoints in a local directory and update the
`model_path` argument in the provided scripts accordingly.

---

## 📁 Data Preparation

Each video should be converted into a folder of RGB frames:

```text
video2img_2fps/
 ├── 0_110/
 │    ├── 0001.png
 │    ├── 0002.png
 │    └── ...
 └── 2_110/
```

During preprocessing, frames are **uniformly sampled to a fixed number (e.g., 20 frames)** per video.

---

## 🧩Functional Modules

### 1️⃣ Feature Extraction

Extract visual features using the pretrained InternVL image encoder:

```bash
cd Visual feature extraction
python extract_features_x.py \
    --video_dir path/to/video_frames \
    --save_dir features\
    --model_path model/
```

The extracted features are stored as `.pt` files and reused for classifier training and inference.

---

### 2️⃣ Classification

Train the lightweight temporal classification head on the extracted VLM features to perform binary detection of three road-rage hazardous behavior categories.

```bash
python train_mlp.py
```

Test Results(Example)

| Method                      | Dangerous Acc | Dangerous F1 | Aggressive Acc | Aggressive F1 | Obstructive Acc | Obstructive F1 |
| --------------------------- | ------------- | ------------ | -------------- | ------------- | --------------- | -------------- |
| **Qwen2.5-VL-32B-Instruct** | **100.0**     | **100.0**    | 64.71          | 50.00         | **82.35**       | **76.92**      |
| Qwen2.5-VL-7B-Instruct      | 100.0         | 100.0        | 70.59          | 44.44         | 70.59           | 66.67          |
| InternVL3-8B                | 88.24         | 93.75        | **70.59**      | **66.67**     | 82.35           | 76.92          |
| LLaVA-onevision-qwen2-7B    | 100.0         | 100.0        | 64.71          | 0.00          | 64.71           | 0.00           |
| LLaVA-NeXT-Video-7B         | 0.00          | 0.00         | 64.71          | 0.00          | 64.71           | 0.00           |


---

## 📬 Data & Resource Availability

For dataset access or additional experimental resources, please contact **roadragereasoning@163.com**.

---

## 📜 License

This project is released under the **MIT License**.

---
