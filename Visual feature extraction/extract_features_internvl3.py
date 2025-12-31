import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor


# =====================================================
# 1. Dataset: one folder corresponds to one video
# =====================================================
class VideoFolderDataset(Dataset):
    """
    Each subfolder under root_dir corresponds to one video.
    Video frames are uniformly sampled to a fixed number.
    """

    def __init__(self, root_dir, num_frames=20, last_n=None):
        self.root_dir = root_dir
        self.num_frames = num_frames

        all_videos = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Optionally use only the last N videos (for debugging or ablation)
        self.video_folders = all_videos[-last_n:] if last_n is not None else all_videos

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]

        frame_paths = sorted(
            glob.glob(os.path.join(video_folder, "*.jpg")) +
            glob.glob(os.path.join(video_folder, "*.png"))
        )

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in {video_folder}")

        # -------- Uniform temporal sampling --------
        if len(frame_paths) < self.num_frames:
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))
        else:
            indices = np.linspace(
                0, len(frame_paths) - 1,
                self.num_frames,
                dtype=int
            )
            frame_paths = [frame_paths[i] for i in indices]

        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        video_id = os.path.basename(video_folder)

        return frames, video_id


# =====================================================
# 2. Load InternVL model (vision encoder only)
# =====================================================
def load_model(model_path, device):
    """
    Load InternVL model and image processor.
    Only the vision encoder is used.
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    image_processor = AutoImageProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model.vision_model.eval()
    return model, image_processor


# =====================================================
# 3. Visual-only video feature extraction
# =====================================================
@torch.no_grad()
def extract_video_feature(frames, model, image_processor, device):
    """
    Args:
        frames: List[PIL.Image], length = T
    Returns:
        video_feat: Tensor of shape [D]
    """

    inputs = image_processor(
        images=frames,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(
        device=device,
        dtype=model.dtype
    )
    # Shape: [T, 3, H, W]

    outputs = model.vision_model(
        pixel_values,
        output_hidden_states=True
    )

    # Last hidden state: [T, N, D]
    feats = outputs.last_hidden_state

    # Token-level average pooling -> [T, D]
    feats = feats.mean(dim=1)

    # Temporal average pooling -> [D]
    video_feat = feats.mean(dim=0)

    return video_feat.cpu()


# =====================================================
# 4. Main feature extraction pipeline
# =====================================================
def main():
    # -------------------------------------------------
    # Anonymous paths (to be replaced by users)
    # -------------------------------------------------
    VIDEO_FRAME_ROOT = "PATH/TO/VIDEO_FRAME_ROOT"
    FEATURE_SAVE_DIR = "PATH/TO/SAVE_FEATURES"
    MODEL_PATH = "PATH/TO/INTERNVL_MODEL"

    os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, image_processor = load_model(MODEL_PATH, device)

    # Build dataset
    dataset = VideoFolderDataset(
        root_dir=VIDEO_FRAME_ROOT,
        num_frames=20
    )

    # Custom collate function to avoid PIL stacking issues
    def collate_fn(batch):
        frames, video_ids = zip(*batch)
        return frames[0], video_ids[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    # Feature extraction loop
    for frames, video_id in tqdm(loader, desc="Extracting InternVL video features"):
        feature = extract_video_feature(
            frames=frames,
            model=model,
            image_processor=image_processor,
            device=device
        )

        torch.save(
            feature,
            os.path.join(FEATURE_SAVE_DIR, f"{video_id}.pt")
        )

    print("InternVL video feature extraction completed successfully.")


if __name__ == "__main__":
    main()
