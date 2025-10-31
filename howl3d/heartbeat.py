from enum import Enum

class HeartbeatType(Enum):
    TaskStart = "start"
    TaskUpdate = "update"
    TaskComplete = "complete"

class Heartbeat:
    def __init__(self, job_id):
        self.job_id = job_id

    def send(self, type, task, msg=None):
        # If heartbeat contains a human-readable message, print it to the console
        if msg: print(f"Job {self.job_id} sent heartbeat from task {task}: {msg}", flush=True)