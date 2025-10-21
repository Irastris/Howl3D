# Howl3D: Free & Open-Source 3D SBS Conversion
__version__ = "0.0.1"

from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from howl3d.video_conversion import VideoConversion
from howl3d.utils.directories import cleanup_directory

if __name__ == "__main__":
    # Load config from disk
    with open("./config.yml") as config_file:
        config = yaml.safe_load(config_file)
    # Add runtime variables into the config
    config["device"] = torch.device("cuda") # Set Torch device to be used for any applicable operations going forward
    config["thread_pool"] = ThreadPoolExecutor(max_workers=config["threads"]) # Construct pool for any multithreaded processes going forward

    # Initialize the video converter
    video_path = "example.mp4" # TODO: Implement a proper argsparse or similar. Assuming this is video file in the same directory as the script during early development.
    video_conversion = VideoConversion(config, video_path)

    # Process the video
    print(f"Processing {video_path}")
    video_conversion.process_video()

    # Cleanup working directory if enabled
    if config["cleanup"]: cleanup_directory(config["working_path"])