import shutil

def ensure_directory(path, cleanup=True):
    if cleanup and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def cleanup_directory(path):
    if path.exists():
        shutil.rmtree(path)