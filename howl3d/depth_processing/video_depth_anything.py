from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.utils.directories import ensure_directory
from thirdparty.video_depth_anything.video_depth import VideoDepthAnything

from functools import partial
print = partial(print, flush=True)

vda_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

class VideoDepthAnythingProcessor(BaseDepthProcessor):
    def __init__(self, config):
        super().__init__(config, "vda_depth_dir")
        self.depth_stats = {"min": float("inf"), "max": float("-inf")}
        self.depths_yaml_path = Path(self.config["working_dir"]) / f"{self.config['vda_depth_dir']}.yaml"

    def get_depth_normalization_params(self, depth):
        self.load_depth_stats()
        d_min = self.depth_stats["min"]
        d_max = self.depth_stats["max"]
        print(f"d_min: {d_min}\nd_max: {d_max}")
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return depth_norm

    def check_yaml_exists(self):
        return not self.depths_yaml_path.exists()

    def compute_depths(self, start_idx, end_idx, model):
        # Load frames from disk
        frames = []
        for i in range(start_idx, end_idx):
            frame_path = self.config["frames_output_path"] / f"frame_{i:06d}.png"
            frame = cv2.imread(str(frame_path))
            frames.append(frame)

        # Convert to numpy array if needed
        if not isinstance(frames, np.ndarray):
            frames = np.stack(frames, axis=0)

        # Process through depth model
        depths, _ = model.infer_video_depth(frames, self.config["video_info"]["framerate"], input_size=580, device="cuda", fp32=False)

        # Update depth statistics
        self.config["depth_stats"]["min"] = min(self.config["depth_stats"]["min"], depths.min())
        self.config["depth_stats"]["max"] = max(self.config["depth_stats"]["max"], depths.max())

        # Save depth maps to disk
        for i, depth_map in enumerate(depths):
            frame_idx = start_idx + i
            depth_path = self.config["depths_output_path"] / f"depth_{frame_idx:06d}.npy"
            np.save(str(depth_path), depth_map)

        # Save depth stats to disk for reruns
        self.save_depth_stats()

        # Cleanup memory
        del frames
        del depths
        torch.cuda.empty_cache()

    def save_depth_stats(self):
        depth_stats = {
            "depth_min": float(self.config["depth_stats"]["min"]),
            "depth_max": float(self.config["depth_stats"]["max"])
        }
        with open(self.depths_yaml_path, "w") as f:
            yaml.dump(depth_stats, f)

    def load_depth_stats(self):
        with open(self.depths_yaml_path) as f:
            depths_yaml = yaml.safe_load(f)
        self.depth_stats["min"] = depths_yaml["depth_min"]
        self.depth_stats["max"] = depths_yaml["depth_max"]

    def process(self):
        # Check if depths are already exported
        if self.should_process("vda_depth_dir", self.check_yaml_exists):
            # Load VideoDepthAnything model
            vda_model = self.config["vda_model"]
            print(f"Loading VideoDepthAnything model, {vda_model} variant")
            video_depth_anything = VideoDepthAnything(**vda_model_configs[vda_model], metric=False)
            video_depth_anything.load_state_dict(torch.load(f"models/video_depth_anything/{vda_model}.pth", map_location="cpu"), strict=True)
            video_depth_anything = video_depth_anything.to("cuda").eval()

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"])

            # Compute depth in batches
            print(f"Computing depths for {self.config['video_info']['frames']} frames in batches of {self.config['vda_batch_size']}")
            for batch_num, start_idx in enumerate(range(0, self.config["video_info"]["frames"], self.config["vda_batch_size"]), 1):
                end_idx = min(start_idx + self.config["vda_batch_size"], self.config["video_info"]["frames"])
                self.compute_depths(start_idx, end_idx, video_depth_anything)

            # Cleanup model from GPU
            del video_depth_anything
            torch.cuda.empty_cache()
        else:
            print("Depths already exported, skipping depth computation")