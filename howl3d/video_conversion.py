import os
from pathlib import Path

import cv2

from howl3d.depth_processing.video_depth_anything import VideoDepthAnythingProcessor
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

    def export_frames(self):
        capture = cv2.VideoCapture(str(self.config["video_path"]))

        frame_count = 0
        while (frame_count + 1) <= self.config["video_info"]["frames"]:
            ret, frame = capture.read()
            if not ret: break
            frame_filename = self.config["frames_output_path"] / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            frame_count += 1

        capture.release()

        return frame_count

    def process_video(self, print_video_info=False):
        if print_video_info:
            print("Video Info:")
            print(f"File Size: {self.config['video_info']['filesize'] / (1024 ** 2):.2f} MB")
            print(f"Dimensions: {self.config['video_info']['width']}x{self.config['video_info']['height']}")
            print(f"Total Frames: {self.config['video_info']['frames']}")
            print(f"Framerate: {self.config['video_info']['framerate']:.2f} fps")
            print(f"Duration: {self.config['video_info']['duration']:.2f} seconds")
            print(f"Codec: {self.config['video_info']['codec']}")

        # Ensure frame output directory exists, cleaning up existing contents if they exist
        ensure_directory(self.config["frames_output_path"], True)

        # Export frames
        frames = self.config["video_info"]["frames"]
        print(f"Exporting {frames} frames from video")
        self.export_frames()

        # Generate depth maps using VideoDepthAnything with batch processing
        print("Running depth processor")
        depth_processor = VideoDepthAnythingProcessor(self.config)
        depth_processor.process_video()

        # Encode depth video
        # output_depth_video = self.config["video_path"].parent / (self.config["video_path"].stem + "_depths" + self.config["video_path"].suffix)
        # print(f"Encoding depth video to {output_depth_video}")
        # depth_processor.encode_video(output_depth_video)
