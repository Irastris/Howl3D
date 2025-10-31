from pathlib import Path

import cv2

from howl3d.heartbeat import Heartbeat
from howl3d.depth_processing.depth_anything_v2 import DepthAnythingV2Processor
from howl3d.depth_processing.depth_pro import DepthProProcessor
from howl3d.depth_processing.distill_any_depth import DistillAnyDepthProcessor
from howl3d.depth_processing.temporal_smoothing import TemporalSmoothingProcessor
from howl3d.depth_processing.video_depth_anything import VideoDepthAnythingProcessor
from howl3d.sbs_processing.stereovision import StereoVisionProcessor
from howl3d.utils import copy_file, ensure_directory

class MediaInfo:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MediaConversion:
    def __init__(self, job):
        self.config = job.config
        self.config["media_path"] = Path(job.media_path)
        self.config["media_info"] = self.get_media_info()
        self.config["working_path"] = Path(self.config["working_dir"])
        self.config["frames_output_path"] = self.config["working_path"] / self.config["frames_dir"]
        self.heartbeat = Heartbeat(job.id)
        self.job = job

    def get_media_info(self):
        if self.config["media_path"].suffix.lower() in [".jpeg", ".jpg", ".png", ".webp"]:
            image = cv2.imread(str(self.config["media_path"]))

            # Extract image properties
            height, width = image.shape[:2]

            return MediaInfo(
                type="image",
                filesize=self.config["media_path"].stat().st_size,
                width=width,
                height=height,
                frames=1
            )
        elif self.config["media_path"].suffix.lower() in [".mkv", ".mp4", ".webm"]:
            capture = cv2.VideoCapture(str(self.config["media_path"]))

            # Extract video properties
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            framerate = capture.get(cv2.CAP_PROP_FPS)
            duration = frames / framerate if framerate > 0 else 0
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).upper()

            capture.release()

            return MediaInfo(
                type="video",
                filesize=self.config["media_path"].stat().st_size,
                width=width,
                height=height,
                frames=frames,
                framerate=framerate,
                duration=duration,
                codec=codec
            )

    def should_export_frames(self):
        if not self.config["frames_output_path"].exists(): return True
        existing_frames = len(list(self.config["frames_output_path"].glob("frame_*.png")))
        return True if existing_frames != self.config["media_info"].frames else False

    def export_frames(self):
        capture = cv2.VideoCapture(str(self.config["media_path"]))
        for i in range(self.config["media_info"].frames):
            _, frame = capture.read()
            frame_filename = self.config["frames_output_path"] / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            self.heartbeat.send(msg=f"Exported frame {i}/{self.config['media_info'].frames}")
        capture.release()

    def process(self):
        # Ensure frame output directory exists
        ensure_directory(self.config["frames_output_path"])

        if self.config["media_info"].type == "image": # Copy single image to frames directory # TODO: Lazy solution to avoid deeper refactoring, will need to address eventually
            copy_file(self.config["media_path"], self.config["frames_output_path"] / "frame_000000.png")
        elif self.config["media_info"].type == "video" and self.should_export_frames(): # Check if frames are already exported
            self.heartbeat.send(msg=f"Exporting {self.config['media_info'].frames} frames from video")
            self.export_frames()
            self.heartbeat.send(msg="Finished exporting frames from video")

        # Generate depth maps
        self.heartbeat.send(msg="Running depth processor")
        if self.config["depth_processor"] == "DepthAnythingV2":
            depth_processor = DepthAnythingV2Processor(self.config)
        elif self.config["depth_processor"] == "DepthPro":
            depth_processor = DepthProProcessor(self.config)
        elif self.config["depth_processor"] == "DistillAnyDepth":
            depth_processor = DistillAnyDepthProcessor(self.config)
        elif self.config["depth_processor"] == "VideoDepthAnything":
            depth_processor = VideoDepthAnythingProcessor(self.config)
        depth_processor.process()
        self.heartbeat.send(msg="Finished processing depth")

        # Temporally smooth depth maps
        # TODO: Implement some form of edge masking so that this doesn't result in an overall blurring of the depths, especially at higher window sizes
        if self.config["media_info"].type == "video" and self.config["enable_temporal_smoothing"]:
            self.heartbeat.send(msg="Running temporal smoothing processor")
            ts_processor = TemporalSmoothingProcessor(self.config)
            ts_processor.process()
            self.heartbeat.send(msg="Finished temporal smoothing")

        # Generate sterescopic images using StereoVision with multithreading
        self.heartbeat.send(msg="Running stereoscopic processor")
        if self.config["stereo_processor"] == "StereoVision":
            stereo_processor = StereoVisionProcessor(self.config, self.job.id)
        stereo_processor.process()
        self.heartbeat.send(msg="Finished stereoscopic processing")

        # Save depth
        self.heartbeat.send(msg="Saving depth")
        output_depth = self.config["media_path"].parent / (self.config["media_path"].stem + "_depths" + ("_ts" if self.config["media_info"].type == "video" and self.config["enable_temporal_smoothing"] else "") + ".mp4")
        depth_processor.save(output_depth)
        self.heartbeat.send(msg="Depth saved")

        # Save SBS
        self.heartbeat.send(msg="Saving SBS")
        output_sbs = self.config["media_path"].parent / (self.config["media_path"].stem + "_sbs" + ".mp4")
        stereo_processor.save(output_sbs)
        self.heartbeat.send(msg="SBS saved")