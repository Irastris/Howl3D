import numpy as np
from pathlib import Path

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.utils.directories import ensure_directory
from tqdm import tqdm

class TemporalSmoothingProcessor(BaseDepthProcessor):
    def __init__(self, config):
        super().__init__(config, self.get_depth_dir_key(config["depth_processor"]))
        self.config["depths_ts_output_path"] = (Path(self.config["working_dir"]) / self.config[self.get_depth_dir_key(self.config["depth_processor"])]) / "ts"

    @staticmethod
    def get_depth_dir_key(depth_processor):
        if depth_processor == "DepthAnythingV2": return "da2_depth_dir"
        elif depth_processor == "DepthPro": return "dp_depth_dir"
        elif depth_processor == "DistillAnyDepth": return "dad_depth_dir"
        elif depth_processor == "VideoDepthAnything": return "vda_depth_dir"

    def should_process(self):
        if not self.config["depths_ts_output_path"].exists(): return True
        return len(list(self.config["depths_ts_output_path"].glob("depth_*.npy"))) != self.media_info.frames

    def smooth_depths(self, depths):
        mode = self.config["ts_mode"]
        window_size = self.config["ts_window_size"]

        smoothed = []
        for i in tqdm(range(len(depths))):
            if mode == "backward":
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
            elif mode == "forward":
                start_idx = i
                end_idx = min(len(depths), i + window_size)
            elif mode == "bidirectional":
                half_window = window_size // 2
                start_idx = max(0, i - half_window)
                end_idx = min(len(depths), i + half_window + 1)

            # Collect frames in the window
            frames_to_average = depths[start_idx:end_idx]

            # Stack and compute mean
            stacked = np.stack(frames_to_average, axis=0)
            smoothed_frame = np.mean(stacked, axis=0)

            smoothed.append(smoothed_frame)

        return smoothed

    def process(self):
        # Check if smoothed depths are already exported
        if self.should_process():
            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_ts_output_path"])

            # Load depths from disk
            depths = []
            for i in range(self.media_info.frames):
                depths.append(np.load(str((Path(self.config["working_dir"]) / (self.config[self.get_depth_dir_key(self.config["depth_processor"])]) / f"depth_{i:06d}.npy"))))

            # Smooth depths
            depths = self.smooth_depths(depths)

            # Save smoothed depth maps to disk
            for i, depth_map in enumerate(depths):
                depth_path = self.config["depths_ts_output_path"] / f"depth_{i:06d}.npy"
                np.save(str(depth_path), depth_map)
        else:
            print("Temporally smoothed depths already exported, skipping smoothed depth computation")