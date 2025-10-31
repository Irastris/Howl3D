from enum import Enum

class HeartbeatType(Enum):
    TaskStart = "start"
    TaskUpdate = "update"
    TaskComplete = "complete"

class Heartbeat:
    def __init__(self, job_id):
        self.job_id = job_id

    def send(self, type, task, msg):
        print(f"Job {self.job_id} sent heartbeat from task {task}: {msg}", flush=True)