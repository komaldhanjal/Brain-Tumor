# Project Overview
This project is an end-to-end Brain Tumor Detection System that uses Machine Learning and Image Processing to detect brain tumors from MRI images.
The system classifies MRI scans into tumor and non-tumor categories and provides predictions through a desktop-based graphical user interface (GUI).
All predictions are also stored in a MySQL database for future analysis and record keeping.


#Features: 

🧠 Brain tumor detection from MRI images
🖼 Image preprocessing (grayscale, resizing, flattening, scaling)
🤖 Pre-trained machine learning model for prediction
🖥 User-friendly Tkinter-based GUI
📂 Train & Test image navigation
🗄 Prediction results stored in MySQL database
📊 Image-wise prediction with real-time results

# Tech Stack

## Programming Language
- Python

## Libraries & Tools
- Scikit-learn
- NumPy
- Pillow (PIL)
- Joblib
- Tkinter
- MySQL

## Concepts Used
- Machine Learning
- Image Processing
- GUI Development
- Database Integration


# Model Details

- Type: Machine Learning Classification Model
- Input: Preprocessed MRI image vectors
- Output: Tumor / No Tumor
- Scaling: Standard Scaler
- Model saved using Joblib


⚙️ How It Works

MRI images are loaded from the training or testing dataset
Images are converted to grayscale and resized
Image data is flattened and scaled
The trained ML model predicts whether a tumor is present
Prediction result is displayed on the GUI
Result is automatically stored in the MySQL database

#Future Enhancements:

Add Deep Learning (CNN) model
Improve accuracy with more data
Web-based deployment using Streamlit
User authentication and reporting dashboard