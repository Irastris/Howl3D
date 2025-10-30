import cv2
import numpy as np
import torch
from tqdm import tqdm

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.utils.directories import ensure_directory
from thirdparty.depth_anything_v2.dpt import DepthAnythingV2

da2_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Adapted from DepthAnythingV2's run.py -- https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py
class DepthAnythingV2Processor(BaseDepthProcessor):
    def __init__(self, config):
        super().__init__(config, "da2_depth_dir")

    def get_depth_normalization_params(self, depth):
        d_min = depth.min()
        d_max = depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return depth_norm

    def compute_depths(self, frame_idx, model):
        frame_path = self.config["frames_output_path"] / f"frame_{frame_idx:06d}.png"

        # Load image
        image = cv2.imread(str(frame_path))

        # Process through depth model
        depth = model.infer_image(image, 518)

        # Save depth map to disk
        depth_path = self.config["depths_output_path"] / f"depth_{frame_idx:06d}.npy"
        np.save(str(depth_path), depth)

    def process(self):
        if self.should_process("da2_depth_dir"):
            # Load DepthAnythingV2 model
            da2_model = self.config["da2_model"]
            print(f"Loading DepthAnythingV2 model, {da2_model} variant")
            depth_anything = DepthAnythingV2(**da2_model_configs[da2_model])
            depth_anything.load_state_dict(torch.load(f"models/depth_anything_v2/{da2_model}.pth", map_location="cpu"), strict=True)
            depth_anything = depth_anything.to("cuda").eval()

            print(f"Computing depths for {self.media_info.frames} frames")

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"])

            # Construct a manually updated progress bar
            if self.media_info.type == "video": pbar = tqdm(range(self.media_info.frames))

            # Compute depth for each frame
            for i in range(self.media_info.frames):
                self.compute_depths(i, depth_anything)
                if self.media_info.type == "video":
                    pbar.update(1)
                    pbar.refresh()

            # Cleanup model from GPU
            del depth_anything
            torch.cuda.empty_cache()
        else:
            print("Depths already exported, skipping depth computation")