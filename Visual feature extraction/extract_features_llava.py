import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq


# =====================================================
# 1. Dataset: One folder corresponds to one video
# =====================================================
class VideoFolderDataset(Dataset):
    """
    Each subfolder under root_dir represents one video.
    Frames inside each folder are uniformly sampled
    to a fixed number of frames.
    """

    def __init__(self, root_dir, num_frames=20):
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

        frame_paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.png"))
        )

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in folder: {folder}")

        # -------- Uniform temporal sampling --------
        if len(frame_paths) < self.num_frames:
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))
        else:
            indices = np.linspace(
                0, len(frame_paths) - 1,
                self.num_frames, dtype=int
            )
            frame_paths = [frame_paths[i] for i in indices]

        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        video_id = os.path.basename(folder)

        return frames, video_id


# =====================================================
# 2. Load vision-language model (vision encoder only)
# =====================================================
def load_model(model_path, device):
    """
    Load a pretrained vision-language model and
    keep only the vision encoder for feature extraction.
    """

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    model.vision_tower.eval()

    return model, processor


# =====================================================
# 3. Visual-only video feature extraction
# =====================================================
@torch.no_grad()
def extract_video_feature(frames, model, processor, device):
    """
    Args:
        frames: List[PIL.Image], length = T
    Returns:
        video_feat: Tensor of shape [D]
    """

    inputs = processor(
        images=frames,
        text=[""] * len(frames),  # dummy text input
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)
    # Shape: [1, T, 3, H, W]

    # Flatten batch and temporal dimensions
    B, T, C, H, W = pixel_values.shape
    pixel_values = pixel_values.view(B * T, C, H, W)
    # Shape: [T, 3, H, W]

    outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True
    )

    # Last hidden state: [T, N, D]
    feats = outputs.hidden_states[-1]

    # Token-level pooling -> [T, D]
    feats = feats.mean(dim=1)

    # Temporal pooling -> [D]
    video_feat = feats.mean(dim=0)

    return video_feat.cpu()


# =====================================================
# 4. Main entry
# =====================================================
def main():
    # -----------------------------
    # Anonymous paths (placeholders)
    # -----------------------------
    VIDEO_ROOT = "/path/to/video_frames_root"
    SAVE_DIR = "/path/to/save_extracted_features"
    MODEL_PATH = "/path/to/pretrained_vlm"

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load model
    # -----------------------------
    model, processor = load_model(MODEL_PATH, device)

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    dataset = VideoFolderDataset(
        root_dir=VIDEO_ROOT,
        num_frames=20
    )

    def collate_fn(batch):
        frames, video_ids = zip(*batch)
        return frames[0], video_ids[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # -----------------------------
    # Feature extraction loop
    # -----------------------------
    for frames, video_id in tqdm(loader, desc="Extracting video features"):
        feat = extract_video_feature(
            frames=frames,
            model=model,
            processor=processor,
            device=device
        )

        torch.save(
            feat,
            os.path.join(SAVE_DIR, f"{video_id}.pt")
        )

    print("All video features have been successfully extracted.")


if __name__ == "__main__":
    main()
