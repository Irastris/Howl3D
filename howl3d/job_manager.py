import hashlib
from enum import Enum
from pathlib import Path

class JobStatus(Enum):
    Queued = 1
    Processing = 2
    Completed = 3
    Failed = 4

class Job:
    def __init__(self, config, media_path):
        self.media_path = Path(media_path)
        self.job_id = self.generate_job_id()
        self.config = self.generate_isolated_config(config)
        self.status = JobStatus.Queued

    def generate_job_id(self):
        hash_input = str(self.media_path).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()[:12]

    def generate_isolated_config(self, config):
        isolated_config = config.copy()
        isolated_config["working_dir"] = str(Path(config["working_dir"]) / f"{self.job_id}")
        return isolated_config

class JobManager:
    def __init__(self, concurrent_jobs=1):
        self.concurrent_jobs = concurrent_jobs

    def submit_job(self, config, media_path):
        job = Job(config, media_path)
        print(f"Job submitted for {job.media_path} with a working directory of {job.config['working_dir']}")