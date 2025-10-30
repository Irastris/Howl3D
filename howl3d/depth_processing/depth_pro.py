import numpy as np
import torch
from tqdm import tqdm

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.utils.directories import ensure_directory
from thirdparty.depth_pro import create_model_and_transforms, load_rgb

# Adapted from DepthPro's run.py -- https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/cli/run.py
class DepthProProcessor(BaseDepthProcessor):
    def __init__(self, config):
        super().__init__(config, "dp_depth_dir")

    def get_depth_normalization_params(self, depth):
        depth = 1 / depth # Invert the map
        d_min = depth.min()
        d_max = depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return depth_norm

    def compute_depths(self, frame_idx, model, transform):
        frame_path = self.config["frames_output_path"] / f"frame_{frame_idx:06d}.png"

        # Load image and focal length from EXIF info if present
        image, _, f_px = load_rgb(frame_path)

        # Run prediction
        prediction = model.infer(transform(image), f_px=f_px)  # If `f_px` is provided, it is used to estimate the final metric depth, otherwise the model estimates `f_px` to compute the depth metricness.

        # Extract depth from the prediction
        depth_map = prediction["depth"].detach().cpu().numpy().squeeze()

        # Save depth map to disk
        depth_path = self.config["depths_output_path"] / f"depth_{frame_idx:06d}.npy"
        np.save(str(depth_path), depth_map)

    def process(self):
        if self.should_process("dp_depth_dir"):
            # Load Depth Pro model
            # print("Loading DepthPro model")
            depth_pro, depth_pro_transform = create_model_and_transforms(device=self.config["device"], precision=torch.half)
            depth_pro.eval()

            # print(f"Computing depths for {self.media_info.frames} frames")

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"])

            # Construct a manually updated progress bar
            if self.media_info.type == "video": pass # pbar = tqdm(range(self.media_info.frames))

            # Compute depth for each frame
            for i in range(self.media_info.frames):
                self.compute_depths(i, depth_pro, depth_pro_transform)
                if self.media_info.type == "video":
                    pass
                    # pbar.update(1)
                    # pbar.refresh()

            # Cleanup model from GPU
            del depth_pro
            torch.cuda.empty_cache()
        else:
            pass # print("Depths already exported, skipping depth computation")