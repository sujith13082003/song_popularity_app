import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained ML model
model = joblib.load("song_popularity_model.pkl")

# Streamlit UI layout
st.set_page_config(page_title="🎵 Song Popularity Predictor")
st.title("🎶 Predict the Popularity of a Song")
st.markdown("Enter the audio features below to predict how popular a song might be.")

# Input sliders for user data
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 60.0, 200.0, 120.0)

# When Predict button is clicked
if st.button("🎯 Predict Popularity"):
    # Prepare input data
    input_features = np.array([[danceability, energy, acousticness, tempo]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Show result
    st.success(f"🔥 Predicted Popularity Score: {prediction:.2f} / 100")

    # Optional: display user inputs
    st.markdown("### 🔍 Input Summary")
    input_df = pd.DataFrame(input_features, columns=["danceability", "energy", "acousticness", "tempo"])
    st.dataframe(input_df)
