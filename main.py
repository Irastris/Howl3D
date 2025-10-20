# Howl3D: Free & Open-Source 3D SBS Conversion
__version__ = "0.0.1"

# Global variables
device = torch.device('cuda') # Set Torch device to be used for any applicable operations going forward
thread_pool = ThreadPoolExecutor(max_workers=8) # Construct pool for potential multithreaded processes

if __name__ == '__main__':
    # Create global config to passing data down with
    config = {}
    # Add variables into the config
    config["DEVICE"] = device
    config["THREAD_POOL"] = thread_pool

    print("Hello, world!")
    
    # Initialize the video converter
    video_path = "example.mp4" # TODO: Implement a proper argsparse or similar. Assuming this is video file in the same directory as the script during early development.
    video_conversion = VideoConversion(config, video_path)

    # Process the video
    video_conversion.process_video()