import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from functools import partial
print = partial(print, flush=True)

class BaseStereoProcessor(ABC):
    def __init__(self, config, sbs_dir_key):
        self.config = config
        self.config["sbs_output_path"] = Path(self.config["working_dir"]) / self.config[sbs_dir_key]

    def should_process(self, addon=None):
        if addon and addon(): return True
        path = self.config["sbs_output_path"]
        if not path.exists(): return True
        return len(list(path.glob("sbs_*.png"))) != self.config["video_info"]["frames"]

    @staticmethod
    def pad_frame(image):
        height, width, channels = image.shape

        # Each eye occupies half the width
        eye_width = width // 2
        eye_height = height

        # Calculate the aspect ratio of each eye
        current_aspect = eye_width / eye_height
        target_aspect = 16 / 9

        # Skip padding if the image is already 16:9
        if abs(current_aspect - target_aspect) < 0.001:
            return image

        if current_aspect < target_aspect:
            # Need to add horizontal padding
            target_eye_width = int(eye_height * target_aspect)
            total_padding = target_eye_width - eye_width

            # Pad each eye separately
            left_pad = total_padding // 2
            right_pad = total_padding - left_pad

            # Create padded image
            new_width = width + 2 * total_padding
            padded = np.zeros((height, new_width, channels), dtype=image.dtype)

            # Copy eyes with padding
            padded[:, left_pad:left_pad + eye_width, :] = image[:, :eye_width, :]
            right_eye_start = target_eye_width + left_pad
            padded[:, right_eye_start:right_eye_start + eye_width, :] = image[:, eye_width:, :]
        else:
            # Need to add vertical padding
            target_eye_height = int(eye_width / target_aspect)
            total_padding = target_eye_height - eye_height

            # Pad each eye separately
            top_pad = total_padding // 2
            bottom_pad = total_padding - top_pad

            # Create padded image
            new_height = height + total_padding
            padded = np.zeros((new_height, width, channels), dtype=image.dtype)

            # Copy eyes with vertical padding
            padded[top_pad:top_pad + height, :, :] = image

        return padded

    def encode_video(self, output_path):
        # Remove file at output path if it exists
        output_path.unlink(missing_ok=True)

        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-r", str(self.config["video_info"]["framerate"]),
            "-i", str(self.config["sbs_output_path"] / "sbs_%06d.png"),
            "-vf", "scale=-1:1080",
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        # Start ffmpeg process
        process = subprocess.run(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {process.stderr}")

    def get_depth_normalization_params(self, frame_idx):
        depths_path = (self.config["depths_ts_output_path"] if self.config["enable_temporal_smoothing"] else self.config["depths_output_path"])
        depth = np.load(str(depths_path / f"depth_{frame_idx:06d}.npy"))

        if self.config["depth_processor"] == "DepthPro":
            depth = 1 / depth  # Invert the map
            d_min = max(1 / self.config["dp_depth_max"], depth.min())
            d_max = min(depth.max(), 1 / self.config["dp_depth_min"])
        elif self.config["depth_processor"] == "VideoDepthAnything":
            d_min = self.config["depth_stats"]["min"]
            d_max = self.config["depth_stats"]["max"]

        return ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    @abstractmethod
    def compute_sbs(self, frame_idx):
        pass

    @abstractmethod
    def process(self):
        pass