import os
import argparse
import cv2
import numpy as np
import torch
import pickle
import sys
from PIL import Image

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.module1_foundations.transformations import ImageTransformer
    from src.module1_foundations.degradations import ImageDegrader
    from src.module1_foundations.restoration import ImageRestorer
    from src.module2_classical.features import FeatureExtractor
    from src.module3_deep.model import TrafficSignNet
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Could not import project modules: {e}")
    print("Ensure you are running this script from the root directory 'RobustTrafficSignVision'.")
    print(f"Current working directory: {os.getcwd()}")
    print("Trying to fix python path...")
    sys.path.append(os.getcwd())
    try:
        from src.module1_foundations.transformations import ImageTransformer
        from src.module1_foundations.degradations import ImageDegrader
        from src.module1_foundations.restoration import ImageRestorer
        from src.module2_classical.features import FeatureExtractor
        from src.module3_deep.model import TrafficSignNet
        print(" [x] Import fix successful!")
    except ImportError:
        print(" [!] Failed again. Exiting.")
        sys.exit(1)

# Classes
CLASSES = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

def load_models():
    models = {}
    print("Loading models...")
    if os.path.exists("models/classical_model.pkl"):
        try:
            with open("models/classical_model.pkl", "rb") as f:
                models['classical'] = pickle.load(f)
            print(" [x] Classical model loaded.")
        except Exception as e:
            print(f" [!] Error loading classical model: {e}")
    else:
        print(" [!] Classical model not found.")

    if os.path.exists("models/deep_model.pth"):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = TrafficSignNet(num_classes=43)
            net.load_state_dict(torch.load("models/deep_model.pth", map_location=device))
            net.to(device)
            net.eval()
            models['deep'] = net
            models['device'] = device
            print(" [x] Deep learning model loaded.")
        except Exception as e:
            print(f" [!] Error loading deep model: {e}")
    else:
        print(" [!] Deep model not found.")
    
    return models

def get_image_from_user():
    while True:
        path = input("\nEnter path to image file (or 'q' to quit): ").strip()
        if path.lower() == 'q':
            return None
        # Remove quotes if user dragged and dropped
        path = path.replace('"', '').replace("'", "")
        
        if os.path.exists(path):
            try:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, path
            except:
                print("Error reading image. Try again.")
        else:
            print("File not found.")

def module1_menu(image, path):
    print("\n--- Module 1: Foundations ---")
    degrader = ImageDegrader()
    restorer = ImageRestorer()
    
    # 1. Degrade
    print("Select Degradation:")
    print("1. None")
    print("2. Gaussian Noise")
    print("3. Salt & Pepper")
    print("4. Motion Blur")
    choice = input("Choice (1-4): ")
    
    processed = image.copy()
    if choice == '2':
        processed = degrader.add_gaussian_noise(processed)
        print("Applied Gaussian Noise.")
    elif choice == '3':
        processed = degrader.add_salt_and_pepper(processed)
        print("Applied Salt & Pepper.")
    elif choice == '4':
        processed = degrader.add_motion_blur(processed)
        print("Applied Motion Blur.")
        
    # 2. Restore
    print("\nSelect Restoration:")
    print("1. None")
    print("2. Gaussian Smoothing")
    print("3. Median Filter")
    r_choice = input("Choice (1-3): ")
    
    if r_choice == '2':
        processed = restorer.gaussian_smoothing(processed)
        print("Applied Gaussian Smoothing.")
    elif r_choice == '3':
        processed = restorer.median_filter(processed)
        print("Applied Median Filter.")
        
    # Metrics
    psnr = cv2.PSNR(image, processed)
    print(f"\nResult PSNR: {psnr:.2f} dB")
    
    save_p = f"result_m1_{os.path.basename(path)}"
    cv2.imwrite(save_p, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {save_p}")

def module2_menu(image, models):
    print("\n--- Module 2: Classical Vision ---")
    if 'classical' not in models:
        print("Error: Classical model not loaded. Train first.")
        return

    extractor = FeatureExtractor()
    features, _ = extractor.extract_hog(image)
    
    # Predict
    features = features.reshape(1, -1)
    print("Extracting HOG features...")
    
    pred = models['classical'].predict(features)[0]
    probs = models['classical'].predict_proba(features)[0]
    conf = np.max(probs)
    
    print(f"\nPREDICTION: {CLASSES.get(pred, str(pred))}")
    print(f"Confidence: {conf*100:.2f}%")

def module3_menu(image, models):
    print("\n--- Module 3: Deep Learning ---")
    if 'deep' not in models:
        print("Error: Deep model not loaded. Train first.")
        return

    net = models['deep']
    device = models['device']
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        outputs = net(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probs.topk(1, dim=1)
        
    pred_idx = top_class.item()
    conf = top_p.item()
    
    print(f"\nPREDICTION: {CLASSES.get(pred_idx, str(pred_idx))}")
    print(f"Confidence: {conf*100:.2f}%")

def main():
    print("===========================================")
    print(" Robust Traffic Sign Recognition System CLI")
    print("===========================================")
    
    models = load_models()
    
    while True:
        res = get_image_from_user()
        if res is None:
            break
        image, path = res
        
        while True:
            print(f"\nSelected Image: {path} ({image.shape})")
            print("Select Module to Run:")
            print("1. Module 1 (Degradations & Restoration)")
            print("2. Module 2 (Classical Prediction)")
            print("3. Module 3 (Deep Learning Prediction)")
            print("4. Load different image")
            print("q. Quit")
            
            choice = input("Choice: ")
            
            if choice == '1':
                module1_menu(image, path)
            elif choice == '2':
                module2_menu(image, models)
            elif choice == '3':
                module3_menu(image, models)
            elif choice == '4':
                break
            elif choice.lower() == 'q':
                return
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    main()
