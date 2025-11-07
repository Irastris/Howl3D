import json

import win32file
import win32pipe

class PipeCommunicator:
    def __init__(self):
        self.pipe_handle = win32pipe.CreateNamedPipe(r"\\.\pipe\\howl3d_pipe", win32pipe.PIPE_ACCESS_DUPLEX, win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT, 1, 65536, 65536, 0, None)

    def read_message(self):
        length_buffer = win32file.ReadFile(self.pipe_handle, 4)
        if length_buffer[0] != 0: return None

        message_length = int.from_bytes(length_buffer[1], 'little')
        message_buffer = win32file.ReadFile(self.pipe_handle, message_length)
        if message_buffer[0] != 0: return None

        message = message_buffer[1].decode('utf-8')
        return json.loads(message)

    def send_heartbeat(self, job_id, message):
        heartbeat_data = {
            "job_id": job_id,
            "message": message
        }

        encoded_message = json.dumps(heartbeat_data).encode('utf-8')
        message_length = len(encoded_message).to_bytes(4, 'little')

        win32file.WriteFile(self.pipe_handle, message_length)
        win32file.WriteFile(self.pipe_handle, encoded_message)

    def close_pipe(self):
        if self.pipe_handle:
            win32file.CloseHandle(self.pipe_handle)
            self.pipe_handle = None