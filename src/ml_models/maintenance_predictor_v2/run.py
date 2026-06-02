import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n==================================================")
    print(f"RUNNING: {script_path}")
    print(f"==================================================")
    
    # Use workspace .venv if available, otherwise fallback to current python interpreter
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_bin = venv_python
    else:
        python_bin = sys.executable
    
    result = subprocess.run([python_bin, script_path], capture_output=False)
    if result.returncode == 0:
        print(f"\n[SUCCESS] Finished running {script_path}")
    else:
        print(f"\n[FAILED] Error running {script_path}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    scripts = [
        "train_rf.py",
        "train.py",
        "train_mlp.py"
    ]
    
    # Check if all scripts exist before running
    for script in scripts:
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)
            
    print("Starting all RUL Model training scripts (v2 - Pipeline CV Edition)...")
    for script in scripts:
        run_script(script)
    print("\n[ALL DONE] All models have been successfully trained and saved!")
