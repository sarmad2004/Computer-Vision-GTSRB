import streamlit as st
import os
import cv2
import numpy as np
import torch
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

from src.module1_foundations.transformations import ImageTransformer
from src.module1_foundations.degradations import ImageDegrader
from src.module1_foundations.restoration import ImageRestorer
from src.module2_classical.features import FeatureExtractor
from src.module3_deep.model import TrafficSignNet
from src.module3_deep.explainability import GradCAM

# --- Config & Setup ---
st.set_page_config(page_title="Modular Vision System", layout="wide", page_icon="🔬")

@st.cache_resource
def load_models():
    models = {}
    # Classical
    if os.path.exists("models/classical_model.pkl"):
        try:
            with open("models/classical_model.pkl", "rb") as f:
                models['classical'] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading classical model: {e}")
    
    # Deep
    if os.path.exists("models/deep_model.pth"):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = TrafficSignNet(num_classes=43)
            # Handle potential state dict mismatch if code changed
            net.load_state_dict(torch.load("models/deep_model.pth", map_location=device))
            net.to(device)
            net.eval()
            models['deep'] = net
            models['device'] = device
        except Exception as e:
            st.error(f"Error loading deep model: {e}")
    return models

models = load_models()

# GTSRB Classes
CLASSES = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

# --- Helper Functions ---
def calculate_metrics(original, processed):
    # PSNR
    psnr = cv2.PSNR(original, processed)
    # SSIM
    grayA = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return psnr, score

# --- UI Layout ---
st.title("🔬 Modular Vision System: Research Pipeline")
st.markdown("Scientific validation of traffic sign recognition across three modular stages.")

# Sidebar
st.sidebar.header("Experimental Controls")
mode = st.sidebar.radio("Select Research Module", 
    ["1. Foundations (Pre-processing)", 
     "2. Classical Vision (Features)", 
     "3. Intelligent Vision (Deep Learning)"])

uploaded_file = st.sidebar.file_uploader("Upload Test Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image_pil)
    
    # Layout: Original Image always visible in sidebar or top expander
    with st.expander("Original Input", expanded=True):
        st.image(image_np, width=200)

    # --- Module 1 ---
    if mode.startswith("1."):
        st.header("Module 1: Image Degradation & Restoration Analysis")
        st.info("Objective: Study the effects of mathematical transformations and noise on image interpretability.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Degradation Simulation")
            deg_type = st.selectbox("Degradation Type", ["None", "Gaussian Noise", "Salt & Pepper", "Motion Blur", "Occlusion"])
            
            degrader = ImageDegrader()
            degraded_img = image_np.copy()
            
            # Tunable Parameters
            if deg_type == "Gaussian Noise":
                sigma = st.slider("Sigma (Intensity)", 0, 100, 25)
                degraded_img = degrader.add_gaussian_noise(degraded_img, sigma=sigma)
            elif deg_type == "Salt & Pepper":
                prob = st.slider("Noise Probability", 0.0, 0.2, 0.02)
                degraded_img = degrader.add_salt_and_pepper(degraded_img, salt_prob=prob, pepper_prob=prob)
            elif deg_type == "Motion Blur":
                k_size = st.slider("Kernel Size", 3, 31, 15, step=2)
                degraded_img = degrader.add_motion_blur(degraded_img, kernel_size=k_size)
            elif deg_type == "Occlusion":
                occ_size = st.slider("Occlusion Box Size", 10, 100, 50)
                degraded_img = degrader.add_occlusion(degraded_img, size=occ_size)
                
            st.image(degraded_img, caption=f"Degraded Input", use_column_width=True)

        with col2:
            st.subheader("2. Restoration Filters")
            rest_type = st.selectbox("Restoration Method", ["None", "Gaussian Smoothing", "Median Filter", "Bilateral Filter", "Non-Local Means"])
            
            restorer = ImageRestorer()
            restored_img = degraded_img.copy()
            
            # Tunable Parameters
            if rest_type == "Gaussian Smoothing":
                k_size_rest = st.slider("Smooth Kernel", 3, 21, 5, step=2)
                restored_img = restorer.gaussian_smoothing(restored_img, kernel_size=k_size_rest)
            elif rest_type == "Median Filter":
                k_size_med = st.slider("Median Kernel", 3, 21, 3, step=2)
                restored_img = restorer.restoration_median_filter(restored_img, kernel_size=k_size_med) if hasattr(restorer, 'restoration_median_filter') else restorer.median_filter(restored_img, kernel_size=k_size_med)
            elif rest_type == "Bilateral Filter":
                d = st.slider("Diameter", 5, 20, 9)
                sigmaColor = st.slider("Sigma Color", 10, 150, 75)
                restored_img = restorer.bilateral_filter(restored_img, d=d, sigmaColor=sigmaColor, sigmaSpace=75)
            elif rest_type == "Non-Local Means":
                st.caption("Note: NLM is computationally intensive.")
                restored_img = restorer.non_local_means_denoising(restored_img)
                
            st.image(restored_img, caption="Restored Output", use_column_width=True)

        if deg_type != "None" or rest_type != "None":
            st.subheader("Quantitative Metrics")
            psnr_val, ssim_val = calculate_metrics(image_np, restored_img)
            m1, m2 = st.columns(2)
            m1.metric("PSNR (Signal-to-Noise)", f"{psnr_val:.2f} dB", delta_color="normal")
            m2.metric("SSIM (Structure)", f"{ssim_val:.4f}", help="1.0 is identical")

    # --- Module 2 ---
    elif mode.startswith("2."):
        st.header("Module 2: Classical Feature Analysis")
        st.info("Objective: Transition from raw pixels to abstract feature descriptors (HOG, LBP).")
        
        feature_method = st.radio("Select Feature Descriptor", ["HOG (Histogram of Oriented Gradients)", "LBP (Local Binary Patterns)"])
        
        extractor = FeatureExtractor()
        
        if "HOG" in feature_method:
            features, viz = extractor.extract_hog(image_np)
            
            # Normalize viz to 0-1 range for Streamlit
            if viz.max() > 0:
                viz = viz.astype(float) / viz.max()
            viz = np.clip(viz, 0.0, 1.0)
            
            st.image(viz, caption="HOG Visualization (Gradient Magnitudes)", width=400, clamp=True)
            
            st.markdown("### Classifier Prediction (SVM)")
            if 'classical' in models:
                # Prediction
                feats_reshaped = features.reshape(1, -1)
                try:
                    pred = models['classical'].predict(feats_reshaped)[0]
                    probs = models['classical'].predict_proba(feats_reshaped)[0]
                    conf = np.max(probs)
                    st.success(f"Class: **{CLASSES.get(pred, str(pred))}**")
                    st.progress(float(conf))
                    st.caption(f"Confidence: {conf*100:.1f}%")
                except:
                    st.warning("Model expects different feature shape or HOG parameters. Ensure training matched current parameters.")
            else:
                st.warning("Classical model not loaded.")
                
        elif "LBP" in feature_method:
            radius = st.slider("LBP Radius", 1, 5, 1)
            n_points = st.slider("LBP Points", 8, 24, 8)
            hist, lbp_img = extractor.extract_lbp(image_np, radius=radius, n_points=n_points)
            
            # Normalize for visualization
            lbp_viz = (lbp_img.astype(float) / lbp_img.max() * 255).astype(np.uint8)
            
            c1, c2 = st.columns(2)
            c1.image(lbp_viz, caption="LBP Texture Map", use_column_width=True)
            
            # Plot Histogram
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.bar(range(len(hist)), hist, color='gray')
            ax.set_title("LBP Histogram (Texture Signature)")
            c2.pyplot(fig)
            
            st.info("LBP captures texture patterns. Flat areas have low variance, edges/corners have specific signatures.")

    # --- Module 3 ---
    elif mode.startswith("3."):
        st.header("Module 3: Intelligent Vision (Deep Learning)")
        st.info("Objective: End-to-end learning with explainability (Grad-CAM).")
        
        if 'deep' not in models:
            st.error("Deep model not found. Please run training script.")
        else:
            net = models['deep']
            device = models['device']
            
            # Preprocessing
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            input_tensor = transform(image_np).unsqueeze(0).to(device)
            
            # Inference
            with st.spinner("Running Neural Network..."):
                outputs = net(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_class = probs.topk(5, dim=1)
                
                # Convert to numpy
                top_p = top_p.cpu().detach().numpy()[0]
                top_class = top_class.cpu().detach().numpy()[0]
                
            # Results
            st.subheader("Classification Results")
            best_class = CLASSES.get(top_class[0], "Unknown")
            st.success(f"Prediction: **{best_class}** ({top_p[0]*100:.2f}%)")
            
            # Charts
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.caption("Top 5 Predictions")
                chart_data = pd.DataFrame({
                    'Class': [CLASSES.get(c, str(c)) for c in top_class],
                    'Confidence': top_p
                })
                # Horizontal Bar
                fig, ax = plt.subplots()
                sns.barplot(data=chart_data, x='Confidence', y='Class', ax=ax, palette='viridis')
                st.pyplot(fig)
                
            with col2:
                st.caption("Explainability (Grad-CAM)")
                # Grad-CAM Generation
                target_layer = net.features[-2] if hasattr(net, 'features') else None 
                # Fallback for ResNet or different arch
                if not target_layer and hasattr(net, 'layer4'): # ResNet
                    target_layer = net.layer4[-1]
                elif hasattr(net, 'features'): # Custom
                    # Heuristic to find last conv
                    for module in reversed(list(net.features)):
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                            break
                            
                if target_layer:
                    grad_cam = GradCAM(net, target_layer)
                    cam, _ = grad_cam.generate_cam(input_tensor, target_class=top_class[0])
                    
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
                    
                    overlay = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)
                    st.image(overlay, caption="Attention Map", use_column_width=True)
                else:
                    st.warning("Could not identify target layer for Grad-CAM automatically.")

else:
    st.write("Please upload an image to begin the scientific evaluation.")
