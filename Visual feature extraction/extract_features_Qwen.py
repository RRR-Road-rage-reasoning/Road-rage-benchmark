import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# =====================================================
# 1. Dataset definition
# =====================================================
class VideoFolderDataset(Dataset):
    """
    Dataset that loads videos represented as folders of frames.
    Each folder corresponds to one video.
    """

    def __init__(self, root_dir, num_frames=20):
        """
        Args:
            root_dir (str): Root directory containing video frame folders
            num_frames (int): Number of frames sampled per video
        """
        self.root_dir = root_dir
        self.num_frames = num_frames

        self.video_folders = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        folder = self.video_folders[idx]
        video_id = os.path.basename(folder)

        # Load all frame paths
        frame_paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.png"))
        )

        # Uniform sampling or padding
        if len(frame_paths) < self.num_frames:
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))
        else:
            indices = np.linspace(
                0, len(frame_paths) - 1, self.num_frames, dtype=int
            )
            frame_paths = [frame_paths[i] for i in indices]

        # Return frame paths instead of images for VLM processing
        return frame_paths, video_id


# =====================================================
# 2. Load pretrained Qwen2.5-VL model
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the pretrained VLM (replace with your local or HF path)
model_path = "PATH/TO/QWEN2.5-VL-7B-INSTRUCT"

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Visual encoder reference
vision_encoder = model.visual
vision_encoder.eval()


# =====================================================
# 3. Video-level feature extraction
# =====================================================
@torch.no_grad()
def extract_video_feature(frame_paths, processor, model):
    """
    Extract a single video-level feature by averaging video token embeddings.

    Args:
        frame_paths (List[str]): Paths to sampled video frames
        processor: HuggingFace processor
        model: Vision-language model

    Returns:
        Tensor of shape [D], where D is the hidden dimension
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                },
                {"type": "text", "text": "Describe the video briefly."}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process visual inputs using official utility
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    # Last hidden states: [B, L, D]
    last_hidden = outputs.hidden_states[-1]

    # Extract video token embeddings
    video_len = inputs["pixel_values_videos"].shape[1]
    video_tokens = last_hidden[:, :video_len, :]  # [1, T, D]

    # Temporal average pooling
    video_feature = video_tokens.mean(dim=1)  # [1, D]

    return video_feature.squeeze(0).cpu()


# =====================================================
# 4. Main feature extraction pipeline
# =====================================================
def main():
    # Root directory containing frame folders
    video_root = "PATH/TO/VIDEO_FRAME_ROOT"

    # Directory to save extracted features
    save_dir = "PATH/TO/SAVE_FEATURES"
    os.makedirs(save_dir, exist_ok=True)

    dataset = VideoFolderDataset(video_root, num_frames=20)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for frame_paths, video_id in tqdm(loader):
        frame_paths = frame_paths[0]  # batch_size = 1
        video_id = video_id[0]

        feature = extract_video_feature(frame_paths, processor, model)
        torch.save(feature, os.path.join(save_dir, f"{video_id}.pt"))

    print("Video features successfully saved.")


if __name__ == "__main__":
    main()
