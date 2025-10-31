import concurrent.futures

import cv2
import numpy as np
from PIL import Image

from howl3d.heartbeat import Heartbeat
from howl3d.sbs_processing.base import BaseStereoProcessor
from howl3d.utils import ensure_directory

# Adapted from ComfyUI-StereoVision -- https://github.com/DrMWeigand/ComfyUI-StereoVision
class StereoVisionProcessor(BaseStereoProcessor):
    def __init__(self, config, job_id):
        super().__init__(config, job_id, "sv_sbs_dir")

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

        # Load corresponding depth map and normalize it
        depth_norm = self.get_depth_normalization_params(frame_idx)

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

        # Pad each eye to a 16:9 aspect ratio if necessary
        sbs_image = self.pad_frame(sbs_image)

        # Save the side-by-side frame
        sbs_frame_path = self.config["sbs_output_path"] / f"sbs_{frame_idx:06d}.png"
        cv2.imwrite(str(sbs_frame_path), cv2.cvtColor(sbs_image, cv2.COLOR_RGB2BGR))

    def process(self):
        # Check if frames are already exported
        if self.should_process():
            self.heartbeat.send(msg=f"Computing {self.media_info.frames} SBS frames on {self.config['threads']} threads")

            # Ensure SBS output directory exists
            ensure_directory(self.config["sbs_output_path"])

            # Submit futures to thread pool
            futures = [self.config["thread_pool"].submit(self.compute_sbs, i) for i in range(self.media_info.frames)]

            # Track completed futures
            for i, _ in enumerate(concurrent.futures.as_completed(futures)):
                self.heartbeat.send(msg=f"Processed SBS frame {i+1}/{self.media_info.frames}")
        else:
            self.heartbeat.send(msg="SBS frames already exported, skipping SBS computation")