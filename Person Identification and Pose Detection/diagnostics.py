# diagnostics.py

import psutil
import os


def get_diagnostics():
    """Returns CPU and RAM usage."""
    cpu_usage = psutil.cpu_percent()

    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the memory usage of the current process in MB
    ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    return cpu_usage, ram_usage
