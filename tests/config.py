from glob import glob
import os

from pytest import fixture, mark

SAVE_DIR = "test_save_dir"

requires_cleanup = mark.usefixtures("cleanup")

@fixture
def cleanup():
    yield
    cleanup_dir(SAVE_DIR)

def cleanup_dir(dir: str):
    for filename in glob(f"{dir}/*"):
        if os.path.isdir(filename):
            cleanup_dir(filename)
        else:
            os.remove(filename)

    if os.path.exists(dir):
        os.rmdir(dir)