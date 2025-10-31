class Heartbeat:
    def __init__(self, job_id):
        self.job_id = job_id

    def send(self, msg):
        print(f"Job {self.job_id} sent heartbeat: {msg}", flush=True)