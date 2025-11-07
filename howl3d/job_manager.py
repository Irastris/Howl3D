import hashlib
import queue
import threading
import time
import traceback
from enum import Enum
from pathlib import Path

from howl3d.media_conversion import MediaConversion

class JobStatus(Enum):
    Queued = 1
    Processing = 2
    Completed = 3
    Failed = 4

class Job:
    def __init__(self, config, media_path):
        self.media_path = Path(media_path)
        self.id = self.generate_id()
        self.config = self.generate_isolated_config(config)
        self.status = JobStatus.Queued

    def generate_id(self):
        hash_input = str(self.media_path).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()[:12]

    def generate_isolated_config(self, config):
        isolated_config = config.copy()
        isolated_config["working_dir"] = str(Path(config["working_dir"]) / f"{self.id}")
        return isolated_config

class JobManager:
    def __init__(self, pipe_communicator):
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.pipe_communicator = pipe_communicator
        self.worker_thread = threading.Thread(target=self.process_jobs, daemon=True).start()
        self.worker_thread_lock = threading.Lock()

    def submit_job(self, config, media_path):
        job = Job(config, media_path)
        with self.worker_thread_lock:
            self.job_queue.put(job)
            self.active_jobs[job.id] = job

    @staticmethod
    def return_exc(job, e):
        return f"\n{traceback.format_exc()}" if job.config["enable_detailed_exceptions"] else e

    def process_jobs(self):
        while True:
            try:
                active_count = sum(1 for job in self.active_jobs.values() if job.status == JobStatus.Processing)
                if active_count < 1: # Number active_count is less than is the number of concurrently allowed jobs
                    try:
                        job = self.job_queue.get(timeout=1)
                        job.status = JobStatus.Processing

                        print(f"Starting job {job.id} for {job.media_path.name}", flush=True)

                        try:
                            media_conversion = MediaConversion(job, self.pipe_communicator)
                            media_conversion.process()
                            job.status = JobStatus.Completed
                            print(f"Job {job.id} completed!", flush=True)
                        except Exception as e:
                            job.status = JobStatus.Failed
                            print(f"Job {job.id} failed: {self.return_exc(job, e)}", flush=True)
                        finally:
                            if job in self.job_queue.queue:
                                self.job_queue.task_done()
                    except queue.Empty:
                        time.sleep(0.1)
                        continue
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in worker thread: {e}")
                time.sleep(1)