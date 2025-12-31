# Road-rage-benchmark
Road rage benchmark focuses on pre-response road rage regulation by reasoning about situational triggers before aggressive behaviors occur. We introduce Road Rage Reasoning tasks, RoadRageBench with rich annotations, extensive VLM evaluations, and a real-time VLM-based reasoning framework to advance proactive in-car safety systems.

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

<img width="2475" height="1120" alt="fig3-framework_01" src="https://github.com/user-attachments/assets/73fa6b98-7ab6-41be-ae7e-b186a324dd09" />

---

## 🎯 Task Definition

The task is formulated as **independent binary classification** for three hazardous driving behavior categories.  
Each category is annotated using **0/1 binary labels**, where `1` indicates the presence of a hazardous behavior.

| Category ID | Hazardous Behavior  | Label = 0 (Normal) | Label = 1 (Hazardous) |
|------------ |---------------------|--------------------|-----------------------|
| 1           | Dangerous Behavior  | Normal driving     | Dangerous driving     |
| 2           | Aggressive Behavior | Normal driving     | Aggressive driving    |
| 3           | Obstructive Behavior| Normal driving     | Obstructive driving   |

Each hazardous behavior category is detected **independently**, while sharing the same visual feature extraction pipeline.

---

## ⚙️ Environment Setup

### Requirements

- Python >= 3.9+
- PyTorch >= 2.x (with CUDA support recommended)
- Tested with CUDA 12.x

### Installation

```bash
pip install -r requirements/x_requirements.txt
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

<img width="945" height="413" alt="image" src="https://github.com/user-attachments/assets/3097b3bd-9bb9-46a8-bdfd-9f86acd3046c" />


---

## 📬 Data & Resource Availability

For dataset access or additional experimental resources, please contact **roadragereasoning@163.com**.

---

## 📜 License

This project is released under the **MIT License**.

---
