import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import pickle
import torch
from PIL import Image, ImageTk
import sys

# Add path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.module1_foundations.degradations import ImageDegrader
from src.module1_foundations.restoration import ImageRestorer
from src.module2_classical.features import FeatureExtractor
from src.module3_deep.model import TrafficSignNet
from src.module3_deep.explainability import GradCAM

# --- THEME CONFIGURATION ---
THEME = {
    "bg_main": "#212121",       # Very Dark Gray
    "bg_sidebar": "#2E2E2E",    # Dark Gray
    "fg_text": "#ECEFF1",       # Off-white
    "accent": "#00E676",        # Bright Green for actions
    "accent_hover": "#00C853",
    "secondary": "#424242",     # Lighter gray for panels
    "highlight": "#29B6F6"      # Light Blue for info
}

class ModernButton(tk.Button):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.default_bg = kwargs.get('bg', THEME['secondary'])
        self.config(relief=tk.FLAT, bd=0, padx=15, pady=8, fg=THEME['fg_text'], cursor="hand2")
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=THEME['accent'])
        self.config(fg='#000000')

    def on_leave(self, e):
        self.config(bg=self.default_bg)
        self.config(fg=THEME['fg_text'])

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robust Vision System [Research Edition]")
        self.root.geometry("1400x900")
        self.root.config(bg=THEME['bg_main'])
        
        # Load logic
        self.models = self.load_models()
        self.degrader = ImageDegrader()
        self.restorer = ImageRestorer()
        self.extractor = FeatureExtractor()
        
        self.current_image = None
        self.processed_image = None
        
        # State variables for sliders
        self.sigma_val = tk.IntVar(value=25)
        self.kernel_val = tk.IntVar(value=5)
        self.noise_prob = tk.DoubleVar(value=0.05)
        
        self.setup_styles()
        self.setup_ui()
        
    def load_models(self):
        models = {}
        # Silence checks for cleaner startup, just verify existence
        if os.path.exists("models/classical_model.pkl"):
            try:
                with open("models/classical_model.pkl", "rb") as f: models['classical'] = pickle.load(f)
            except: pass
        if os.path.exists("models/deep_model.pth"):
            try:
                net = TrafficSignNet(num_classes=43)
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                net.load_state_dict(torch.load("models/deep_model.pth", map_location=dev))
                net.to(dev).eval()
                models['deep'] = net; models['device'] = dev
            except: pass
        return models

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure TFrame, TLabel, TButton defaults for dark mode where possible
        style.configure("Dark.TFrame", background=THEME['bg_main'])
        style.configure("Sidebar.TFrame", background=THEME['bg_sidebar'])
        
        # Custom Scale (Slider) - Tkinter scale is easier to style directly than ttk.Scale often
        
    def setup_ui(self):
        # 1. Header
        header = tk.Frame(self.root, bg=THEME['bg_sidebar'], height=60, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)
        
        lbl_title = tk.Label(header, text="🚦 ROBUST TRAFFIC SIGN RECOGNITION", 
                             bg=THEME['bg_sidebar'], fg=THEME['accent'], 
                             font=("Segoe UI", 16, "bold"))
        lbl_title.pack(side=tk.LEFT, padx=20)
        
        btn_upload = ModernButton(header, text="📂 Upload Test Image", command=self.upload_image, bg=THEME['secondary'])
        btn_upload.pack(side=tk.RIGHT, padx=20)

        # 2. Sidebar
        sidebar = tk.Frame(self.root, bg=THEME['bg_sidebar'], width=350)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Module Select
        lbl_mod = tk.Label(sidebar, text="SELECT MODULE", bg=THEME['bg_sidebar'], fg="#AAAAAA", font=("Segoe UI", 10))
        lbl_mod.pack(anchor="w", padx=20, pady=(20, 5))
        
        self.mod_var = tk.StringVar(value="Module 1")
        
        for mod in ["Module 1: Foundations", "Module 2: Classical Features", "Module 3: Deep Learning"]:
            rb = tk.Radiobutton(sidebar, text=mod, variable=self.mod_var, value=mod, 
                                bg=THEME['bg_sidebar'], fg=THEME['fg_text'], selectcolor=THEME['bg_sidebar'],
                                activebackground=THEME['bg_sidebar'], activeforeground=THEME['accent'],
                                indicatoron=0, height=2, bd=0, command=self.refresh_controls,
                                font=("Segoe UI", 11))
            rb.pack(fill=tk.X, padx=10, pady=2)

        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X, pady=15, padx=20) # Divider

        # Dynamic Controls Area
        self.controls = tk.Frame(sidebar, bg=THEME['bg_sidebar'])
        self.controls.pack(fill=tk.BOTH, expand=True, padx=20)

        # 3. Main Content
        content = tk.Frame(self.root, bg=THEME['bg_main'])
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Image Container
        img_frame = tk.Frame(content, bg=THEME['bg_main'])
        img_frame.pack(fill=tk.BOTH, expand=True, pady=20, padx=20)
        
        # Grid for Side-by-Side
        self.panel_orig = tk.Label(img_frame, bg="#000000", text="Waiting for Input...", fg="#555")
        self.panel_orig.place(rely=0.0, relx=0.0, relwidth=0.5, relheight=0.7)
        
        self.panel_res = tk.Label(img_frame, bg="#000000", text="Output View", fg="#555")
        self.panel_res.place(rely=0.0, relx=0.5, relwidth=0.5, relheight=0.7)
        
        # Captions
        tk.Label(img_frame, text="Original Input", bg=THEME['bg_main'], fg="white").place(rely=0.71, relx=0.2)
        tk.Label(img_frame, text="Processed / Result", bg=THEME['bg_main'], fg=THEME['accent']).place(rely=0.71, relx=0.7)
        
        # Result Log Area
        log_frame = tk.Frame(content, bg=THEME['secondary'], height=200)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        log_frame.pack_propagate(False)
        
        tk.Label(log_frame, text="ANALYSIS METRICS & LOGS", bg=THEME['secondary'], fg="#AAAAAA", font=("Consolas", 10)).pack(anchor="w", padx=10, pady=5)
        
        self.txt_log = tk.Text(log_frame, bg="#111", fg=THEME['accent'], font=("Consolas", 11), bd=0, insertbackground="white")
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.refresh_controls()

    def create_slider(self, parent, label, var, from_, to_):
        tk.Label(parent, text=label, bg=THEME['bg_sidebar'], fg=THEME['fg_text']).pack(anchor="w", pady=(10, 0))
        s = tk.Scale(parent, variable=var, from_=from_, to=to_, orient=tk.HORIZONTAL, 
                     bg=THEME['bg_sidebar'], fg=THEME['fg_text'], troughcolor="#444", 
                     highlightthickness=0, bd=0, activebackground=THEME['accent'])
        s.pack(fill=tk.X)
        return s

    def refresh_controls(self):
        for w in self.controls.winfo_children(): w.destroy()
        
        mod = self.mod_var.get()
        if mod == "Module 1: Foundations":
            tk.Label(self.controls, text="DEGRADATIONS", bg=THEME['bg_sidebar'], fg=THEME['highlight'], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=10)
            
            self.deg_var = tk.StringVar(value="None")
            opts = ["None", "Gaussian Noise", "Salt & Pepper", "Motion Blur", "Occlusion"]
            om = tk.OptionMenu(self.controls, self.deg_var, *opts)
            om.config(bg=THEME['secondary'], fg="white", bd=0, highlightthickness=0)
            om["menu"].config(bg=THEME['secondary'], fg="white")
            om.pack(fill=tk.X, pady=5)
            
            # Sliders (Dynamic Visibility could be added, but stacking is fine for now)
            self.create_slider(self.controls, "Noise Intensity (Sigma)", self.sigma_val, 0, 100)
            
            tk.Label(self.controls, text="RESTORATION", bg=THEME['bg_sidebar'], fg=THEME['highlight'], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(20, 10))
            self.res_var = tk.StringVar(value="None")
            opts_res = ["None", "Gaussian Smoothing", "Median Filter", "Non-Local Means"]
            om2 = tk.OptionMenu(self.controls, self.res_var, *opts_res)
            om2.config(bg=THEME['secondary'], fg="white", bd=0, highlightthickness=0)
            om2["menu"].config(bg=THEME['secondary'], fg="white")
            om2.pack(fill=tk.X, pady=5)

            self.create_slider(self.controls, "Filter Kernel Size", self.kernel_val, 3, 21)

            ModernButton(self.controls, text="⚡ Apply Pipeline", command=self.run_module_1, bg=THEME['accent']).pack(fill=tk.X, pady=20)

        elif mod == "Module 2: Classical Features":
             tk.Label(self.controls, text="FEATURE EXTRACTION", bg=THEME['bg_sidebar'], fg=THEME['highlight'], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=10)
             
             ModernButton(self.controls, text="👁 View HOG Features", command=self.run_m2_hog).pack(fill=tk.X, pady=5)
             ModernButton(self.controls, text="🕸 View LBP Texture", command=self.run_m2_lbp).pack(fill=tk.X, pady=5)
             
             tk.Label(self.controls, text="CLASSIFICATION", bg=THEME['bg_sidebar'], fg=THEME['highlight'], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(20, 10))
             ModernButton(self.controls, text="🔮 Predict (SVM)", command=self.run_m2_predict).pack(fill=tk.X, pady=5)

        elif mod == "Module 3: Deep Learning":
             tk.Label(self.controls, text="INTELLIGENT VISION", bg=THEME['bg_sidebar'], fg=THEME['highlight'], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=10)
             
             ModernButton(self.controls, text="🧠 Run Deep Inference", command=self.run_m3_infer).pack(fill=tk.X, pady=5)
             ModernButton(self.controls, text="🔥 Explain (Grad-CAM)", command=self.run_m3_gradcam, bg="#FF5722").pack(fill=tk.X, pady=(20, 5))

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        
        # Reset
        self.original_image_cv = cv2.imread(path)
        self.original_image_cv = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2RGB)
        self.current_image = self.original_image_cv.copy()
        
        self.display_img(self.original_image_cv, self.panel_orig)
        self.display_img(self.original_image_cv, self.panel_res) # Setup placeholder
        
        self.log(f"Loaded: {os.path.basename(path)} | Resolution: {self.original_image_cv.shape[:2]}")

    def display_img(self, img_arr, label_widget):
        # Resize keeping aspect ratio
        h, w = img_arr.shape[:2]
        container_h = 500 # Approx from layout
        container_w = 600
        
        # Simple resize logic
        scale = min(container_h/h, container_w/w)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(img_arr, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        label_widget.config(image=tk_img, text="", bg="black")
        label_widget.image = tk_img 

    def log(self, text):
        self.txt_log.insert(tk.END, f">> {text}\n")
        self.txt_log.see(tk.END)

    # --- RUNNERS ---
    def run_module_1(self):
        if self.original_image_cv is None: self.log("Error: No image loaded."); return
        
        # Get Params
        img = self.original_image_cv.copy()
        deg = self.deg_var.get()
        res = self.res_var.get()
        sigma = self.sigma_val.get()
        k = self.kernel_val.get()
        if k % 2 == 0: k += 1 # Ensure odd
        
        self.log(f"Applying: {deg} -> {res} (Sigma={sigma}, K={k})")
        
        # Degrade
        if deg == "Gaussian Noise": img = self.degrader.add_gaussian_noise(img, sigma=sigma)
        elif deg == "Salt & Pepper": img = self.degrader.add_salt_and_pepper(img, salt_prob=0.05)
        elif deg == "Motion Blur": img = self.degrader.add_motion_blur(img, kernel_size=k)
        elif deg == "Occlusion": img = self.degrader.add_occlusion(img)
        
        # Restore
        if res == "Gaussian Smoothing": img = self.restorer.gaussian_smoothing(img, kernel_size=k)
        elif res == "Median Filter": img = self.restorer.median_filter(img, kernel_size=k)
        elif res == "Non-Local Means": 
            self.log("Running NLM (Wait...)...")
            self.root.update()
            img = self.restorer.non_local_means_denoising(img)
        
        self.display_img(img, self.panel_res)
        
        # Metric
        psnr = cv2.PSNR(self.original_image_cv, img)
        self.log(f"Result PSNR: {psnr:.2f} dB")

    def run_m2_hog(self):
        if self.original_image_cv is None: return
        _, viz = self.extractor.extract_hog(self.original_image_cv)
        # Normalize for viz
        if viz.max() > 0: viz = viz / viz.max() * 255
        viz = viz.astype(np.uint8)
        self.display_img(viz, self.panel_res)
        self.log("Visualizing HOG Features (Gradient Orientation)")

    def run_m2_lbp(self):
        if self.original_image_cv is None: return
        _, lbp = self.extractor.extract_lbp(self.original_image_cv)
        if lbp.max() > 0: lbp = lbp / lbp.max() * 255
        lbp = lbp.astype(np.uint8)
        self.display_img(lbp, self.panel_res)
        self.log("Visualizing Local Binary Patterns (Texture)")

    def run_m2_predict(self):
        if self.original_image_cv is None: return
        if 'classical' not in self.models: self.log("Model not found. Train first."); return
        
        feats, _ = self.extractor.extract_hog(self.original_image_cv)
        feats = feats.reshape(1, -1)
        pred = self.models['classical'].predict(feats)[0]
        
        # Classes Map (Shortened for demo or imported)
        CLASSES = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons' }
        
        label = CLASSES.get(pred, str(pred))
        self.log(f"SVM Prediction: {label}")

    def run_m3_infer(self):
        if self.original_image_cv is None: return
        if 'deep' not in self.models: self.log("Deep Model not found."); return
        
        net = self.models['deep']
        dev = self.models['device']
        
        from torchvision import transforms
        t = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)), 
                                transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        inp = t(self.original_image_cv).unsqueeze(0).to(dev)
        
        with torch.no_grad():
            pop = torch.nn.functional.softmax(net(inp), dim=1)
            p, c = pop.topk(1, dim=1)
            
        CLASSES = self.get_classes()
        self.log(f"Deep Prediction: {CLASSES.get(c.item(), str(c.item()))} (Conf: {p.item()*100:.1f}%)")

    def run_m3_gradcam(self):
        if self.original_image_cv is None: return
        if 'deep' not in self.models: self.log("Deep Model not found."); return
        
        net = self.models['deep']
        dev = self.models['device']
        from torchvision import transforms
        t = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)), 
                                transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        inp = t(self.original_image_cv).unsqueeze(0).to(dev)
        
        target = None
        if hasattr(net, 'features'): target = net.features[-2]
        
        if target:
            gc = GradCAM(net, target)
            cam, _ = gc.generate_cam(inp)
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (self.original_image_cv.shape[1], self.original_image_cv.shape[0]))
            over = cv2.addWeighted(self.original_image_cv, 0.5, heatmap, 0.5, 0)
            self.display_img(over, self.panel_res)
            self.log("Grad-CAM overlay generated.")

    def get_classes(self):
         return { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
