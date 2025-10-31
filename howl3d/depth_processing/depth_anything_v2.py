import cv2
import numpy as np
import torch

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.heartbeat import HeartbeatType
from howl3d.utils import ensure_directory
from thirdparty.depth_anything_v2.dpt import DepthAnythingV2

da2_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Adapted from DepthAnythingV2's run.py -- https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py
class DepthAnythingV2Processor(BaseDepthProcessor):
    def __init__(self, config, job_id):
        super().__init__(config, job_id, "da2_depth_dir")

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
        self.heartbeat.send(type=HeartbeatType.TaskStart, task="depth_processor", msg="Running depth processor")

        if self.should_process("da2_depth_dir"):
            # Load DepthAnythingV2 model
            da2_model = self.config["da2_model"]
            depth_anything = DepthAnythingV2(**da2_model_configs[da2_model])
            depth_anything.load_state_dict(torch.load(f"models/depth_anything_v2/{da2_model}.pth", map_location="cpu"), strict=True)
            depth_anything = depth_anything.to("cuda").eval()

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"])

            # Compute depth for each frame
            for i in range(self.media_info.frames):
                self.compute_depths(i, depth_anything)
                self.heartbeat.send(type=HeartbeatType.TaskUpdate, task="depth_processor", msg=f"Processed frame {i+1}/{self.media_info.frames}")

            # Cleanup model from GPU
            del depth_anything
            torch.cuda.empty_cache()

            self.heartbeat.send(type=HeartbeatType.TaskComplete, task="depth_processor", msg="Finished processing depth")
        else:
            self.heartbeat.send(type=HeartbeatType.TaskComplete, task="depth_processor", msg="Depths already processed, skipping")