import streamlit as st
import pickle
import pandas as pd


# Load the model, scaler, and PCA
with open("rainfall_prediction_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
pca = data["pca"]
feature_names = data["feature_names"]

# Streamlit UI
st.title("🌦️ Rainfall Prediction System")
st.write("Enter the weather conditions below to predict whether it will rain or not.")

# Input fields
input_values = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0, format="%.2f")
    input_values.append(value)

if st.button("Predict Rainfall"):
    # Create DataFrame from user input
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Scale and reduce dimensions
    scaled_input = scaler.transform(input_df)
    pca_input = pca.transform(scaled_input)

    # Make prediction
    prediction = model.predict(pca_input)[0]

    # Show result
    if prediction == 1:
        st.success("🌧️ Prediction: Rainfall expected.")
    else:
        st.info("☀️ Prediction: No Rainfall expected.")
