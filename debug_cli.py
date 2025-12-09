import sys
import os

print("1. Starting Debug...")
try:
    import cv2
    print(" [x] cv2 imported")
except ImportError as e:
    print(f" [!] cv2 failed: {e}")

try:
    import torch
    print(" [x] torch imported")
except ImportError as e:
    print(f" [!] torch failed: {e}")

try:
    import torchvision
    print(" [x] torchvision imported")
except ImportError as e:
    print(f" [!] torchvision failed: {e}")

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.module1_foundations.transformations import ImageTransformer
    print(" [x] Local imports successful")
except ImportError as e:
    print(f" [!] Local import failed: {e}")

print("Debug Check Complete.")
