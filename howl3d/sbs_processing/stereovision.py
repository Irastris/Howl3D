import concurrent.futures
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from howl3d.utils.directories import ensure_directory

# Adapted from ComfyUI-StereoVision -- https://github.com/DrMWeigand/ComfyUI-StereoVision
class StereoVisionProcessor:
    def __init__(self, config):
        self.config = config
        self.config["sbs_output_path"] = Path(self.config["working_dir"]) / self.config["sv_sbs_dir"]

    def encode_video(self, output_path):
        # Remove file at output path if it exists
        output_path.unlink(missing_ok=True)

        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-r", str(self.config["video_info"]["framerate"]),
            "-i", str(self.config["sbs_output_path"] / "sbs_%06d.png"),
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

    def process_frame(self, frame_idx):
    def should_compute_sbs(self):
        if not self.config["sbs_output_path"].exists(): return True
        existing_frames = len(list(self.config["sbs_output_path"].glob("sbs_*.png")))
        return True if existing_frames != self.config["video_info"]["frames"] else False

    def compute_sbs(self, frame_idx):
        # Load base frame and get its dimensions
        frame_path = self.config["frames_output_path"] / f"frame_{frame_idx:06d}.png"
        image = Image.open(str(frame_path))
        width, height = image.size

        # Create an empty image for the side-by-side result
        sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        depth_scaling = self.config["sv_depth_scale"] / width

        # Fill the base images
        image_array = np.array(image)
        sbs_image[:, :width] = image_array
        sbs_image[:, width:] = image_array

        # Load corresponding depth map and normalize to 0-255 range using global depth stats
        d_min = self.config["depth_stats"]["min"]
        d_max = self.config["depth_stats"]["max"]
        depth = np.load(str(self.config["depths_output_path"] / f"depth_{frame_idx:06d}.npy"))
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Calculate pixel shifts
        pixel_shifts = (depth_norm * depth_scaling).astype(int)

        # Create meshgrid for coordinates
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Calculate new x coordinates for right eye
        new_x_coords = np.clip(x_coords + pixel_shifts, 0, width - 1)

        # Create mask for valid shifts
        valid_mask = new_x_coords < width

        # Apply shifting
        for shift in range(11):
            shifted_coords = np.clip(new_x_coords + shift, 0, width - 1)
            sbs_image[y_coords[valid_mask], shifted_coords[valid_mask]] = image_array[y_coords[valid_mask], x_coords[valid_mask]]

        # Save the side-by-side frame
        sbs_frame_path = self.config["sbs_output_path"] / f"sbs_{frame_idx:06d}.png"
        cv2.imwrite(str(sbs_frame_path), cv2.cvtColor(sbs_image, cv2.COLOR_RGB2BGR))

        return

    def process(self):
        # Check if frames are already exported
        if self.should_compute_sbs():
            print(f"Computing {self.config['video_info']['frames']} SBS frames on {self.config['threads']} threads")

            # Ensure SBS output directory exists
            ensure_directory(self.config["sbs_output_path"], True)

            # Construct a manually updated progress bar
            pbar = tqdm(range(self.config["video_info"]["frames"]))

            # Submit futures to thread pool
            futures = [self.config["thread_pool"].submit(self.compute_sbs, i) for i in
                       range(self.config["video_info"]["frames"])]
            for _ in concurrent.futures.as_completed(futures):
                # Update the progress bar each time a future completes
                pbar.update(1)
                pbar.refresh()
        else:
            print("SBS frames already exported, skipping SBS computation")