# Howl3D: Free & Open-Source 3D SBS Conversion
__version__ = "0.0.1"

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from howl3d.job_manager import JobManager

def handle_stdin(config, job_manager):
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            job_manager.submit_job(config, line.strip())
        except Exception as e:
            print(f"Error processing stdin: {e}")
            break

if __name__ == "__main__":
    # Load config from disk
    with open("./config.yml") as config_file:
        config = yaml.safe_load(config_file)
    # Add runtime variables into the config
    config["device"] = torch.device("cuda") # Set Torch device to be used for any applicable operations going forward
    config["thread_pool"] = ThreadPoolExecutor(max_workers=config["threads"]) # Construct pool for any multithreaded processes going forward

    # Construct the job manager
    job_manager = JobManager(config["concurrent_jobs"])

    # Process stdin input for job submissions until interrupted
    print("Pipe or enter paths to media for processing:")
    stdin_thread = threading.Thread(target=handle_stdin, args=(config, job_manager), daemon=True)
    stdin_thread.start()

    try:
        while stdin_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass