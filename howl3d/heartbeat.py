from enum import Enum

class HeartbeatType(Enum):
    TaskStart = "start"
    TaskUpdate = "update"
    TaskComplete = "complete"

class Heartbeat:
    def __init__(self, job_id, pipe_communicator=None):
        self.job_id = job_id
        self.pipe_communicator = pipe_communicator

    def send(self, type, task, msg=None):
        self.pipe_communicator.send_heartbeat(job_id=self.job_id, heartbeat_type=type.value, task=task, message=msg)

        # If heartbeat contains a human-readable message, print it to the console
        if msg: print(f"Job {self.job_id} sent heartbeat from task {task}: {msg}", flush=True)