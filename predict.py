import streamlit as st
import requests
import numpy as np
import pickle
from datetime import datetime
from dotenv import load_dotenv
import os
from pytz import timezone

# Load environment variables
load_dotenv()
API_KEY = 'ca1d76dde41cf661ff6009df2c27b08f'

# Load model & scaler
model = pickle.load(open('aqi_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🌍 AQI Predictor | Current + Next 5 Hours Forecast with ML")

mode = st.radio("Choose Mode", ["City", "Coordinates"])

# Timezone for IST
ist = timezone('Asia/Kolkata')

# Functions
def get_latlon(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    response = requests.get(url).json()
    if response:
        return response[0]['lat'], response[0]['lon']
    return None, None

def get_current_pollutants(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url).json()
    if 'list' in response:
        return response['list'][0]['components']
    return None

def get_forecast_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url).json()
    if 'list' in response:
        return response['list']
    return None

# Input Section
components = None
forecast = None
location_name = ""

if mode == "City":
    city = st.text_input("Enter City Name")
    if st.button("Get AQI"):
        lat, lon = get_latlon(city)
        if lat:
            location_name = city
            st.success(f"📍 {city} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            components = get_current_pollutants(lat, lon)
            forecast = get_forecast_aqi(lat, lon)
        else:
            st.error("City not found. Please check the spelling or try a nearby location.")

elif mode == "Coordinates":
    lat = st.number_input("Enter Latitude", format="%.4f")
    lon = st.number_input("Enter Longitude", format="%.4f")
    if st.button("Get AQI"):
        location_name = f"Coordinates ({lat:.4f}, {lon:.4f})"
        components = get_current_pollutants(lat, lon)
        forecast = get_forecast_aqi(lat, lon)

# Current AQI Display
if components:
    st.subheader("📊 Current Pollutant Concentrations (µg/m³):")
    st.json(components)

    input_data = np.array([[components.get(k, 0) for k in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]])
    scaled_data = scaler.transform(input_data)
    current_prediction = model.predict(scaled_data)[0]

    current_time = datetime.now(ist)
    st.header(f"✅ AQI for {location_name}")
    st.success(f"Now ({current_time.strftime('%H:%M:%S')} IST): ML Predicted AQI = {current_prediction:.2f}")

# Forecast AQI Display using ML Model
if forecast:
    st.subheader("🕒 AQI Forecast (Next 5 Hours, ML Predicted):")

    current_timestamp = datetime.now().timestamp()
    upcoming_forecasts = [entry for entry in forecast if entry['dt'] > current_timestamp][:5]

    if not upcoming_forecasts:
        st.warning("No forecast data available for the next 5 hours.")
    else:
        for entry in upcoming_forecasts:
            timestamp = datetime.fromtimestamp(entry['dt'], ist)
            pollutants = entry['components']

            # ML Prediction for forecast pollutants
            future_data = np.array([[pollutants.get(k, 0) for k in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]])
            future_scaled = scaler.transform(future_data)
            future_aqi_pred = model.predict(future_scaled)[0]

            st.write(f"**{timestamp.strftime('%H:%M')} IST** ➡️ ML Predicted AQI = {future_aqi_pred:.2f}")
            with st.expander("View Forecast Pollutant Breakdown"):
                st.json(pollutants)
