# Howl3D: Free & Open-Source 3D SBS Conversion
__version__ = "0.0.1"

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from howl3d.job_manager import JobManager
from howl3d.pipe_communicator import PipeCommunicator

def handle_pipe_communication(config, job_manager, pipe_communicator):
    while True:
        try:
            message = pipe_communicator.read_message()
            if not message:
                time.sleep(0.1)
                continue
            job_manager.submit_job(config, message)
        except Exception as e:
            if e.args[0] == 536: continue # Ignore exceptions from pipe not yet being read by parent process
            print(f"Error processing pipe message: {e}")
            break

if __name__ == "__main__":
    # Load config from disk
    with open("./config.yml") as config_file:
        config = yaml.safe_load(config_file)
    # Add runtime variables into the config
    config["device"] = torch.device("cuda") # Set Torch device to be used for any applicable operations going forward
    config["thread_pool"] = ThreadPoolExecutor(max_workers=config["threads"]) # Construct pool for any multithreaded processes going forward

    # Construct the pipe communicator
    pipe_communicator = PipeCommunicator()

    # Construct the job manager with pipe communicator
    job_manager = JobManager(pipe_communicator)

    # Process pipe for job submissions until interrupted
    print("Waiting for job submissions...")
    pipe_thread = threading.Thread(target=handle_pipe_communication, args=(config, job_manager, pipe_communicator), daemon=True)
    pipe_thread.start()

    try:
        while pipe_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pipe_communicator.close_pipe()