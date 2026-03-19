import subprocess, sys, os, site

def fix():
    # Step 1: Uninstall GUI opencv
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python'], 
                   capture_output=True)
    
    # Step 2: Install headless
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'opencv-python-headless==4.8.1.78', '--quiet', '--force-reinstall'],
                   capture_output=True)
    
    print("opencv fix applied")

if __name__ == '__main__':
    fix()