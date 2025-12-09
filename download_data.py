import kagglehub
import shutil
import os

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    
    print("Path to dataset files:", path)
    
    # Define target directory
    target_dir = os.path.abspath("dataset")
    
    # Check if target exists, create if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move contents
    print(f"Moving files to {target_dir}...")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
    print("Dataset setup complete.")
    print("Contents of dataset folder:")
    print(os.listdir(target_dir))

if __name__ == "__main__":
    download_dataset()
