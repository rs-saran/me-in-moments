# Me in Moments 📸  

**Me in Moments** is a fun and intuitive app that helps you find yourself in event photos scattered across multiple devices or folders. Using advanced face recognition technology, this app compares your reference photo with a collection of images and identifies matches based on similarity.  

Try it [here](https://meinmoments.streamlit.app/)

Read about the implementation details [here](https://rs-saran.github.io/projects/20241201_me_in_moments/)

### 🎯 **Features**  
- **Facial Similarity Search:** Upload your photo and search through event photos for matching faces.  
- **Adjustable Similarity Threshold:** Fine-tune the matching sensitivity to get accurate results.  
- **Download Matches:** Easily download all identified images in a single click.  
- **Friendly Interface:** Simplified and interactive user experience built with Streamlit.  

### 🛠️ **Technologies Used**  
- **DeepFace:** For facial recognition and similarity computation.  
- **OpenCV:** For image processing and manipulation.  
- **Streamlit:** For building an interactive web interface.  

### 🚀 **How to Run the App**  
1. Clone this repository:  
   ```bash
   git clone https://github.com/rs-saran/me-in-moments.git
   cd me-in-moments
   ```  
2. **Create a virtual environment**:
   - For **Windows**:
     ```bash
     python -m venv mim_env
     ```
   - For **macOS/Linux**:
     ```bash
     python3 -m venv mim_env
     ```

3. **Activate the virtual environment**:
   - For **Windows**:
     ```bash
     .\mim_env\Scripts\activate
     ```
   - For **macOS/Linux**:
     ```bash
     source mim_env/bin/activate
     ```

3. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Run the app:  
   ```bash
   streamlit run me_in_moments.py
   ```  
5. Open the app in your browser and start finding yourself in moments!


### 📃 **License**  
This project is licensed under the [MIT License](LICENSE).  

