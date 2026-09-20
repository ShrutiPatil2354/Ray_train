import os


def ensure_output_directory(output_dir: str) -> str:
    absolute_dir = os.path.abspath(output_dir)
    os.makedirs(absolute_dir, exist_ok=True)
    return absolute_dir
