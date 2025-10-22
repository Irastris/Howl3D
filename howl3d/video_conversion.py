import os
from pathlib import Path

import cv2
from tqdm import tqdm

from howl3d.depth_processing.depth_pro import DepthProProcessor
from howl3d.depth_processing.video_depth_anything import VideoDepthAnythingProcessor
from howl3d.sbs_processing.stereovision import StereoVisionProcessor
from howl3d.utils.directories import ensure_directory

class VideoConversion:
    def __init__(self, config, video_path):
        self.config = config
        self.config["video_name"] = video_path
        self.config["video_path"] = Path(video_path)
        self.config["video_info"] = self.get_video_info()
        self.config["working_path"] = Path(self.config["working_dir"])
        self.config["frames_output_path"] = Path(self.config["working_dir"]) / self.config["frames_dir"]

    def get_video_info(self):
        capture = cv2.VideoCapture(str(self.config["video_path"]))

        # Extract video properties, calculate duration, get codec
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        framerate = capture.get(cv2.CAP_PROP_FPS)
        duration = frames / framerate
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).upper()

        # Build dictionary
        video_info = {
            "filesize": os.path.getsize(self.config["video_path"]),
            "width": width,
            "height": height,
            "frames": frames,
            "framerate": framerate,
            "duration": duration,
            "codec": codec
        }

        capture.release()

        return video_info

    def should_export_frames(self):
        if not self.config["frames_output_path"].exists(): return True
        existing_frames = len(list(self.config["frames_output_path"].glob("frame_*.png")))
        return True if existing_frames != self.config["video_info"]["frames"] else False

    def export_frames(self):
        capture = cv2.VideoCapture(str(self.config["video_path"]))
        for i in tqdm(range(self.config["video_info"]["frames"])):
            _, frame = capture.read()
            frame_filename = self.config["frames_output_path"] / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
        capture.release()

    def process(self):
        # Check if frames are already exported
        if self.should_export_frames():
            # Ensure frame output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["frames_output_path"], True)

            # Export frames
            frames = self.config["video_info"]["frames"]
            print(f"Exporting {frames} frames from video")
            self.export_frames()
        else:
            print("Frames already exported, skipping frame extraction")

        # Generate depth maps using VideoDepthAnything with batch processing
        print("Running depth processor")
        if self.config["depth_processor"] == "DepthPro":
            depth_processor = DepthProProcessor(self.config)
        elif self.config["depth_processor"] == "VideoDepthAnything":
            depth_processor = VideoDepthAnythingProcessor(self.config)
        depth_processor.process()

        # Generate sterescopic images using StereoVision with multithreading
        print("Running stereoscopy processor")
        stereo_processor = StereoVisionProcessor(self.config)
        stereo_processor.process()

        # Encode depth video
        output_depth_video = self.config["video_path"].parent / (self.config["video_path"].stem + "_depths" + self.config["video_path"].suffix)
        print(f"Encoding depth video to {output_depth_video}")
        depth_processor.encode_video(output_depth_video)

        # Encode SBS video
        output_sbs_video = self.config["video_path"].parent / (self.config["video_path"].stem + "_sbs" + self.config["video_path"].suffix)
        print(f"Encoding SBS video to {output_sbs_video}")
        stereo_processor.encode_video(output_sbs_video)