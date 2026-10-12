"""Entry point forwarding to main MindBloom app."""
import runpy
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")