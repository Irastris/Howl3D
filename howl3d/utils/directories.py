import shutil

def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)

def cleanup_directory(path):
    if path.exists(): shutil.rmtree(path)