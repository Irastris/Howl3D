import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from thirdparty.video_depth_anything.video_depth import VideoDepthAnything

vda_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

class VideoDepthAnythingProcessor:
    def __init__(self, config):
        self.config = config
        self.config["depth_output_path"] = Path(self.config["working_dir"]) / self.config["vda_depth_dir"]
        self.config["depth_stats"] = {"min": float("inf"), "max": float("-inf")}

    def encode_video(self, output_path):
        # Remove file at output path if it exists
        output_path.unlink(missing_ok=True)

        # Get dimensions from first depth frame, do this instead of pulling from video_info in case I implement resizing later
        first_depth = np.load(str(self.config["depth_output_path"] / "depth_000000.npy"))
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

        # Stream normalized depth frames to ffmpeg
        d_min = self.config["depth_stats"]["min"]
        d_max = self.config["depth_stats"]["max"]

        for i in range(self.config["video_info"]["frames"]):
            depth = np.load(str(self.config["depth_output_path"] / f"depth_{i:06d}.npy"))
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            process.stdin.write(depth_norm.tobytes())

        process.stdin.close()

        # Wait for process to finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")

    def process_frame_batch(self, start_idx, end_idx, model):
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
        depths, _ = model.infer_video_depth(frames, self.config["video_info"]["framerate"], input_size=518, device="cuda", fp32=False)

        # Update depth statistics
        self.config["depth_stats"]["min"] = min(self.config["depth_stats"]["min"], depths.min())
        self.config["depth_stats"]["max"] = max(self.config["depth_stats"]["max"], depths.max())

        # Save depth maps to disk
        for i, depth_map in enumerate(depths):
            frame_idx = start_idx + i
            depth_path = self.config["depth_output_path"] / f"depth_{frame_idx:06d}.npy"
            np.save(str(depth_path), depth_map)

        # Cleanup memory
        del frames
        del depths
        torch.cuda.empty_cache()

    def ensure_depth_output_dir(self, cleanup=True):
        if cleanup and self.config["depth_output_path"].exists():
            shutil.rmtree(self.config["depth_output_path"])
        self.config["depth_output_path"].mkdir(parents=True, exist_ok=True)

    def cleanup_frame_output_directory(self):
        if self.config["depth_output_path"].exists():
            shutil.rmtree(self.config["depth_output_path"])

    def process_video(self):
        # Load VideoDepthAnything model
        vda_model = self.config["vda_model"]
        print(f"Loading VideoDepthAnything model, {vda_model} variant")
        video_depth_anything = VideoDepthAnything(**vda_model_configs[vda_model], metric=False)
        video_depth_anything.load_state_dict(torch.load(f"models/video_depth_anything/{vda_model}.pth", map_location="cpu"), strict=True)
        video_depth_anything = video_depth_anything.to("cuda").eval()

        # Ensure depth output directory exists
        self.ensure_depth_output_dir()

        # Compute depth in batches
        print(f"Computing depths for {self.config['video_info']['frames']} frames in batches of {self.config['vda_batch_size']}")
        for batch_num, start_idx in enumerate(range(0, self.config["video_info"]["frames"], self.config["vda_batch_size"]), 1):
            end_idx = min(start_idx + self.config["vda_batch_size"], self.config["video_info"]["frames"])
            self.process_frame_batch(start_idx, end_idx, video_depth_anything)

        # Create final video from saved depth frames
        output_video = self.config["video_path"].parent / (self.config["video_path"].stem + "_depths" + self.config["video_path"].suffix)
        print(f"Encoding depth video to {output_video}")
        self.encode_video(output_video)

        # Cleanup model from GPU
        del video_depth_anything
        torch.cuda.empty_cache()