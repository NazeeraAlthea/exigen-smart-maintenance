import os
import sys

# Get absolute path to models/whisper-medium
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
whisper_dir = os.path.join(BASE_DIR, "models", "whisper-medium")

print(f"Checking/Downloading Whisper Medium model to: {whisper_dir}...")

try:
    from faster_whisper import download_model
    path = download_model("medium", output_dir=whisper_dir)
    print(f"Success! Model whisper-medium ready at: {path}")
except ImportError:
    print("Error: 'faster-whisper' package is not installed. Please run pip install first.")
    sys.exit(1)
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
