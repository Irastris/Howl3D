class Heartbeat:
    def __init__(self, job_id):
        self.job_id = job_id

    def send(self, msg):
        print(msg, flush=True)