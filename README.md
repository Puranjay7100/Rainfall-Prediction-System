
# 🌦️ Rainfall Prediction System

An interactive machine learning-based web application designed to predict rainfall based on real-time weather conditions. Built using **Streamlit**, this system allows users to input weather-related parameters and receive a prediction on whether it will rain.

---

## 🚀 Features

- Easy-to-use web interface with input fields for weather parameters
- Real-time prediction of rainfall using a trained ML model
- Preprocessing pipeline includes scaling and PCA for better accuracy and performance
- Model trained and serialized for efficient and quick deployment
- Friendly and informative UI with clear result messages, including rainfall visualization using emojis

---

## 🧠 Machine Learning Workflow

This system uses a robust machine learning pipeline comprising:
- **Standard Scaler**: Normalizes input features for better model performance
- **PCA (Principal Component Analysis)**: Reduces the dimensionality of data, improving generalization and reducing overfitting
- **Classifier Model**: Predicts rainfall likelihood based on transformed input features

All these components are packaged and stored in `rainfall_prediction_model.pkl` using Python's `pickle` module for ease of use and deployment.

---

## 🔧 Tech Stack

- **Python**
- **Streamlit** - For building the interactive web app
- **Pandas** - For data manipulation and input handling
- **Scikit-learn** - For ML modeling, feature scaling, and PCA transformation

---

## 📂 Files in the Project

- `app.py` - Main Streamlit application script responsible for UI and prediction logic
- `rainfall_prediction_model.pkl` - Pickled file containing:
  - Trained ML model
  - Scaler for input normalization
  - PCA object for dimensionality reduction
  - Feature names required for prediction inputs
- `project.ipynb` - Jupyter Notebook containing:
  - Dataset exploration and cleaning
  - Outlier detection and handling
  - Feature scaling and PCA transformation
  - Model training, tuning, and evaluation
  - Final model export using `pickle`

---

## 🔄 How to Run

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn
   ```

2. **Ensure files are placed in the same directory**:
   - `app.py`
   - `rainfall_prediction_model.pkl`

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

Upon running, a local web server will launch in your browser where you can input weather conditions and receive rainfall predictions instantly.

---

## 📅 Input Format

The system will prompt users to input values for specific weather features, such as:
- Temperature
- Humidity
- Wind Speed
- Atmospheric Pressure
- Dew Point, etc.

*(Exact features depend on the contents of `feature_names` within the pickle file.)*

---

## 📢 Output

Based on the processed input, the system returns one of the following predictions:
- **🌧️ Rainfall expected**
- **☀️ No Rainfall expected**

This binary classification helps users quickly understand the likelihood of rain in a given scenario.

---

## 📊 Notebook Insights (`project.ipynb`)

The notebook serves as the backbone of model development and includes:
- Data Cleaning & Preprocessing routines
- Handling and removal of Outliers
- Feature Scaling & PCA dimensionality reduction
- Model Comparison using metrics such as accuracy, precision, and recall
- Selection of best-performing model
- Final Model Export using `pickle` for deployment in `app.py`

---


## 📅 Note

- Ensure `rainfall_prediction_model.pkl` is present in the root directory while running the app.
- This project is excellent for demonstrating machine learning deployment using Streamlit and can be extended with more sophisticated models or real-time weather APIs.

---

## 🚀 Future Improvements

- Integrate advanced deep learning models such as LSTM or CNN for time-series weather prediction
- Incorporate live weather data using external APIs
- Enhance the user interface with charts, graphs, or map-based visualizations
- Provide model explainability with SHAP or LIME for educational purposes

---

Made with ❤️ for practical machine learning projects and weather enthusiasts.

