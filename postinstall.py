"""
Patches ultralytics to not import opencv-python (GUI) at startup.
Run once after pip install to replace cv2 import with headless version.
"""
import subprocess, sys

def fix():
    # Uninstall GUI opencv, install headless
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python'], check=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'opencv-python-headless==4.8.1.78', '--quiet'], check=False)
    print("opencv-python replaced with headless version")

if __name__ == '__main__':
    fix()