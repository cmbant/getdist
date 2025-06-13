"""
Runtime hook for PyInstaller to fix multiprocessing issues.
This script is executed before the app starts to ensure multiprocessing works correctly.
"""

import multiprocessing
import multiprocessing.popen_spawn_win32
import sys

# Ensure we're using the correct multiprocessing implementation
multiprocessing.freeze_support()

# Force the use of spawn method on Windows
if sys.platform == "win32":
    multiprocessing.set_start_method("spawn", force=True)
