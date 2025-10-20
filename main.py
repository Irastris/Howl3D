# Howl3D: Free & Open-Source 3D SBS Conversion
__version__ = "0.0.1"

import torch
import yaml
from concurrent.futures import ThreadPoolExecutor
from howl3d.video_conversion import VideoConversion

# Global variables
device = torch.device('cuda') # Set Torch device to be used for any applicable operations going forward
thread_pool = ThreadPoolExecutor(max_workers=8) # Construct pool for potential multithreaded processes

if __name__ == '__main__':
    # Load config from disk
    with open("./config.yml") as config_file:
        config = yaml.safe_load(config_file)
    # Add runtime variables into the config
    config["device"] = device
    config["thread_pool"] = thread_pool

    # Initialize the video converter
    video_path = "example.mp4" # TODO: Implement a proper argsparse or similar. Assuming this is video file in the same directory as the script during early development.
    video_conversion = VideoConversion(config, video_path)

    # Process the video
    video_conversion.process_video()