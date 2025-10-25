import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from howl3d.utils.directories import ensure_directory
from thirdparty.depth_pro import create_model_and_transforms, load_rgb

from functools import partial
print = partial(print, flush=True)

# Adapted from DepthPro's run.py -- https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/cli/run.py
class DepthProProcessor:
    def __init__(self, config):
        self.config = config
        self.config["depths_output_path"] = Path(self.config["working_dir"]) / self.config["dp_depth_dir"]
        self.config["depth_stats"] = {"min": float("inf"), "max": float("-inf")}

    def encode_video(self, output_path):
        # Remove file at output path if it exists
        output_path.unlink(missing_ok=True)

        # Get dimensions from first depth frame
        first_depth = np.load(str(self.config["depths_output_path"] / "depth_000000.npy"))
        height, width = first_depth.shape

        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "gray",
            "-r", str(self.config["video_info"]["framerate"]),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        # Start ffmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Normalize each depth map before piping it to the ffmpeg process
        for i in range(self.config["video_info"]["frames"]):
            depth = np.load(str(self.config["depths_output_path"] / f"depth_{i:06d}.npy"))
            # Invert the map
            depth = 1 / depth
            d_min = max(1 / self.config["dp_depth_max"], depth.min())
            d_max = min(depth.max(), 1 / self.config["dp_depth_min"])
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            process.stdin.write(depth_norm.tobytes())

        process.stdin.close()

        # Wait for process to finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")

    def should_compute_depths(self):
        if not self.config["depths_output_path"].exists(): return True
        existing_depths = len(list(self.config["depths_output_path"].glob("depth_*.npy")))
        return True if existing_depths != self.config["video_info"]["frames"] else False

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
        if self.should_compute_depths():
            # Load Depth Pro model
            print("Loading DepthPro model")
            depth_pro, depth_pro_transform = create_model_and_transforms(device=self.config["device"], precision=torch.half)
            depth_pro.eval()

            print(f"Computing depths for {self.config['video_info']['frames']} frames")

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"], True)

            # Construct a manually updated progress bar
            pbar = tqdm(range(self.config["video_info"]["frames"]))

            # Compute depth for each frame
            for i in range(self.config["video_info"]["frames"]):
                self.compute_depths(i, depth_pro, depth_pro_transform)
                pbar.update(1)
                pbar.refresh()

            # Cleanup model from GPU
            del depth_pro
            torch.cuda.empty_cache()
        else:
            print("Depths already exported, skipping depth computation")