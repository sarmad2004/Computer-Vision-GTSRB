# How to Run the Project

## Option 1: Visual Studio Code (Local Windows/Mac/Linux)
This is the recommended way as it runs faster on your local machine.

### Prerequisites
- Install [Python 3.8+](https://www.python.org/downloads/)
- Install [Visual Studio Code](https://code.visualstudio.com/)

### Steps
1. **Open the Project**:
   - Launch VS Code.
   - Go to **File > Open Folder**.
   - Select the `RobustTrafficSignVision` folder.

2. **Open Terminal**:
   - Press `Ctrl + ~` (tilde) to open the integrated terminal.
   - Or go to **Terminal > New Terminal**.

3. **Install Dependencies**:
   Copy and paste this command into the terminal:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Data**:
   This script will fetch the dataset from Kaggle automatically.
   ```bash
   python download_data.py
   ```

5. **Run the App**:
   
   **Option A: Desktop GUI (Dark Mode)** - *Recommended*
   ```bash
   python gui.py
   ```
   
   **Option B: Web Interface**
   ```bash
   streamlit run app.py
   ```
   
   **Option C: Terminal CLI**
   ```bash
   python cli.py
   ```

---

## Option 2: Google Colab (Cloud)
Since Colab runs in the cloud, we need a special "tunnel" to see the graphical interface (Streamlit App).

### Steps
1. **Upload Files**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Create a New Notebook.
   - On the left sidebar, click the **Folder icon** (Files).
   - Drag and drop the specific files (`app.py`, `requirements.txt`, `download_data.py`, `run_training.py`) and the `src` folder into the files area.
   - *Alternative*: You can zip the whole project on your PC, upload the zip, and unzip it in Colab with `!unzip project.zip`.

2. **Run these commands in a Colab Cell**:
   Copy the following code into a code cell and run it.

   ```python
   # 1. Install Dependencies
   !pip install -r requirements.txt
   !pip install pyngrok  # Needed to show the GUI from Colab

   # 2. Download Data
   !python download_data.py

   # 3. Train Models (If you didn't upload the 'models' folder)
   !python run_training.py

   # 4. Run Streamlit with Tunnel
   from pyngrok import ngrok

   # Terminate open tunnels if any
   ngrok.kill()

   # Set your auth token (Sign up at ngrok.com for free)
   # NGROK_AUTH_TOKEN = "YOUR_TOKEN_HERE" 
   # ngrok.set_auth_token(NGROK_AUTH_TOKEN) 

   # Open a customized tunnel to the streamlit port 8501
   public_url = ngrok.connect(8501).public_url
   print(f"🚀 Streamlit App is live at: {public_url}")

   # Run Streamlit in the background
   !streamlit run app.py &>/dev/null&
   ```

3. **Click the Link**:
   - Detailed output will show a link (e.g., `http://xxxx.ngrok.io`). Click it to view your app.
