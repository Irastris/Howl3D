import os
import cv2

def get_video_info(video_path):
    filesize = os.path.getsize(video_path)

    capture = cv2.VideoCapture(video_path)

    # Extract video properties, calculate duration, get codec
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    framerate = capture.get(cv2.CAP_PROP_FPS)
    duration = frames / framerate
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Build dictionary
    video_info = {
        'name': video_path,
        'filesize': filesize,
        'width': width,
        'height': height,
        'frames': frames,
        'framerate': framerate,
        'duration': duration,
        'codec': codec
    }

    capture.release()

    return video_info

class VideoConversion:
    def __init__(self, config, video_path):
        self.config = config
        self.video_path = video_path
        self.video_info = {}

    def process_video(self):
        self.video_info = get_video_info(self.video_path)
