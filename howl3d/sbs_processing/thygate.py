import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from numba import njit, prange
from tqdm import tqdm

from howl3d.utils.directories import ensure_directory

from functools import partial
print = partial(print, flush=True)

# Adapted from stable-diffusion-webui-depthmap-script's stereoimage_generation.py -- https://github.com/thygate/stable-diffusion-webui-depthmap-script/blob/main/src/stereoimage_generation.py
class ThyGateProcessor:
    def __init__(self, config):
        self.config = config
        self.config["sbs_output_path"] = Path(self.config["working_dir"]) / self.config["tg_sbs_dir"]

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

    @staticmethod
    @njit(parallel=True)
    def apply_stereo_divergence(frame, depth, fill_method, divergence, separation, offset_exponent):
        EPSILON = 1e-7
        PIXEL_HALF_WIDTH = 0.45 if fill_method == "polylines_sharp" else 0.0

        h, w, c = frame.shape
        derived_image = np.zeros_like(frame)
        for row in prange(h):
            pt = np.zeros((5 + 2 * w, 3), dtype=np.float_)
            pt_end: int = 0
            pt[pt_end] = [-1.0 * w, 0.0, 0.0]
            pt_end += 1
            for col in range(0, w):
                coord_d = (depth[row][col] ** offset_exponent) * divergence
                coord_x = col + 0.5 + coord_d + separation
                if PIXEL_HALF_WIDTH < EPSILON:
                    pt[pt_end] = [coord_x, abs(coord_d), col]
                    pt_end += 1
                else:
                    pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                    pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                    pt_end += 2
            pt[pt_end] = [2.0 * w, 0.0, w - 1]
            pt_end += 1

            sg_end: int = pt_end - 1
            sg = np.zeros((sg_end, 6), dtype=np.float_)
            for i in range(sg_end):
                sg[i] += np.concatenate((pt[i], pt[i + 1]))
            for i in range(1, sg_end):
                u = i - 1
                while pt[u][0] > pt[u + 1][0] and 0 <= u:
                    pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                    sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                    u -= 1

            csg = np.zeros((5 * int(abs(divergence)) + 25, 6), dtype=np.float_)
            csg_end: int = 0
            sg_pointer: int = 0
            pt_i: int = 0
            for col in range(w):
                color = np.full(c, 0.5, dtype=np.float_)
                while pt[pt_i][0] < col:
                    pt_i += 1
                pt_i -= 1
                while pt[pt_i][0] < col + 1:
                    coord_from = max(col, pt[pt_i][0]) + EPSILON
                    coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
                    significance = coord_to - coord_from
                    coord_center = coord_from + 0.5 * significance

                    while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
                        csg[csg_end] = sg[sg_pointer]
                        sg_pointer += 1
                        csg_end += 1
                    csg_i = 0
                    while csg_i < csg_end:
                        if csg[csg_i][3] < coord_center:
                            csg[csg_i] = csg[csg_end - 1]
                            csg_end -= 1
                        else:
                            csg_i += 1
                    best_csg_i: int = 0
                    if csg_end != 1:
                        best_csg_closeness: float = -EPSILON
                        for csg_i in range(csg_end):
                            ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
                            closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
                            if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
                                best_csg_closeness = closeness
                                best_csg_i = csg_i
                    col_l: int = int(csg[best_csg_i][2] + EPSILON)
                    col_r: int = int(csg[best_csg_i][5] + EPSILON)
                    if col_l == col_r:
                        color += frame[row][col_l] * significance
                    else:
                        ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
                        color += (frame[row][col_l] * (1.0 - ip_k) +
                                  frame[row][col_r] * ip_k
                                  ) * significance
                    pt_i += 1
                derived_image[row][col] = np.asarray(color, dtype=np.uint8)
        return derived_image

    def should_compute_sbs(self):
        if not self.config["sbs_output_path"].exists(): return True
        existing_frames = len(list(self.config["sbs_output_path"].glob("sbs_*.png")))
        return True if existing_frames != self.config["video_info"]["frames"] else False

    def compute_sbs(self, frame_idx):
        # Load base frame and get its dimensions
        frame_path = self.config["frames_output_path"] / f"frame_{frame_idx:06d}.png"
        image = Image.open(str(frame_path))
        width, height = image.size

        # Load corresponding depth map and normalize to 0-255 range, using global depth stats if available
        depths_path = self.config["depths_ts_output_path"] if self.config["enable_temporal_smoothing"] else self.config["depths_output_path"]
        depth = np.load(str(depths_path / f"depth_{frame_idx:06d}.npy"))
        if self.config["depth_processor"] == "DepthPro":
            depth = 1 / depth # Invert the map
            d_min = max(1 / self.config["dp_depth_max"], depth.min())
            d_max = min(depth.max(), 1 / self.config["dp_depth_min"])
            depth_norm = (depth - d_min) / (d_max - d_min)
        elif self.config["depth_processor"] == "VideoDepthAnything":
            d_min = self.config["depth_stats"]["min"]
            d_max = self.config["depth_stats"]["max"]
            depth_norm = (depth - d_min) / (d_max - d_min)

        # Convert the image to a numpy array
        image_array = np.asarray(image)

        # Calculate the left and right eyes
        left_eye = image_array
        right_eye = image_array
        if self.config["tg_balance"] > 0.001:
            divergence = ((1 * self.config["tg_divergence"] * self.config["tg_balance"]) / 100.0) * width
            separation = ((-1 * self.config["tg_separation"]) / 100.0) * width
            left_eye = self.apply_stereo_divergence(image_array, depth_norm, self.config["tg_fill_method"], divergence, separation, self.config["tg_offset_exponent"])
        if self.config["tg_balance"] < 0.999:
            divergence = ((-1 * self.config["tg_divergence"] * (1 - self.config["tg_balance"])) / 100.0) * width
            separation = (self.config["tg_separation"] / 100.0) * width
            right_eye = self.apply_stereo_divergence(image_array, depth_norm, self.config["tg_fill_method"], divergence, separation, self.config["tg_offset_exponent"])

        # Combine the left and right eyes
        sbs_image = np.hstack([left_eye, right_eye])

        # Pad each eye to a 16:9 aspect ratio if necessary
        sbs_image = self.pad_frame(sbs_image)

        # Save the side-by-side frame
        sbs_frame_path = self.config["sbs_output_path"] / f"sbs_{frame_idx:06d}.png"
        cv2.imwrite(str(sbs_frame_path), cv2.cvtColor(sbs_image, cv2.COLOR_RGB2BGR))

    def process(self):
        # Check if frames are already exported
        if self.should_compute_sbs():
            print(f"Computing {self.config['video_info']['frames']} SBS frames with numba parallelization")

            # Ensure SBS output directory exists
            ensure_directory(self.config["sbs_output_path"])

            # Construct a manually updated progress bar
            pbar = tqdm(range(self.config["video_info"]["frames"]))

            for i in range(self.config["video_info"]["frames"]):
                self.compute_sbs(i)
                # Update the progress bar each time a future completes
                pbar.update(1)
                pbar.refresh()
        else:
            print("SBS frames already exported, skipping SBS computation")