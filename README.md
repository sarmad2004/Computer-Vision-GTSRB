# Robust Traffic Sign Recognition System

This is a modular computer vision system designed for the "Modular Vision System" project requirements.

## Project Structure
- `dataset/`: Contains the GTSRB dataset (downloaded automatically).
- `src/`: Source code for the 3 modules.
    - `module1_foundations/`: Image transformations & restoration.
    - `module2_classical/`: Feature extraction (HOG) & SVM classification.
    - `module3_deep/`: CNN model & Grad-CAM explainability.
- `models/`: Stores trained models (`classical_model.pkl`, `deep_model.pth`).
- `app.py`: Streamlit GUI for interactive demonstration.
- `run_training.py`: Script to retrain models.
- `download_data.py`: Script to download dataset from Kaggle.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data** (if not already done):
   ```bash
   python download_data.py
   ```

3. **Train Models**:
   (Optional, models may be pre-trained)
   ```bash
   python run_training.py
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Modules

### Module 1: Foundations
- Upload an image.
- Apply Noise (Gaussian, Salt & Pepper) or Occlusion.
- Apply Restoration (Gaussian Blur, Median Filter, NLM).
- View effect on image quality.

### Module 2: Classical Vision
- Extracts HOG features.
- Classifies using a Support Vector Machine (SVM).
- Shows HOG visualization.

### Module 3: Deep Learning
- Uses a custom CNN (or ResNet).
- Classifies the image.
- **Explainability**: Shows Grad-CAM heatmap to visualize what the model is looking at.
