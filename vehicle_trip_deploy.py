import streamlit as st
import joblib
import pandas as pd

model = joblib.load("vehicle_trip_model.pkl")
st.title("🚗 Vehicle Trip Duration Predictor")
st.markdown("Enter the trip details below:")

distance = st.number_input("Trip Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
hour = st.slider("Pickup Hour (0-23)", 0, 23, 8)
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

day_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

if st.button("Predict Duration"):
    input_df = pd.DataFrame([[distance, hour, day_index, is_weekend]], columns=["distance_km", "hour", "day_of_week", "is_weekend"])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Trip Duration: {prediction:.2f} minutes")
