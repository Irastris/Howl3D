import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

class BaseDepthProcessor(ABC):
    def __init__(self, config, depth_dir_key):
        self.config = config
        self.config["depths_output_path"] = Path(self.config["working_dir"]) / self.config[depth_dir_key]
        # Shortcuts
        self.media_info = self.config["media_info"]

    def should_process(self, dir_key, addon=None):
        if addon and addon(): return True
        path = Path(self.config["working_dir"]) / self.config[dir_key]
        if not path.exists(): return True
        return len(list(path.glob("depth_*.npy"))) != self.media_info.frames

    def get_depth_normalization_params(self, depth):
        raise NotImplementedError("Subclasses must implement depth normalization")

    def encode_video(self, output_path):
        # Remove file at output path if it exists
        output_path.unlink(missing_ok=True)

        # Get dimensions from first depth frame, necessary to provide to ffmpeg for stdin input.
        depths_path = self.config["depths_ts_output_path"] if self.config["enable_temporal_smoothing"] else self.config["depths_output_path"]
        first_depth = np.load(str(depths_path / "depth_000000.npy"))
        height, width = first_depth.shape

        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "gray",
            "-r", str(self.media_info.framerate),
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

        # Normalize and write each depth map
        for i in range(self.media_info.frames):
            depth = np.load(str(depths_path / f"depth_{i:06d}.npy"))
            depth_norm = self.get_depth_normalization_params(depth)
            process.stdin.write(depth_norm.tobytes())

        process.stdin.close()

        # Wait for process to finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")

    @abstractmethod
    def process(self):
        pass