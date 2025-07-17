import streamlit as st
import requests
import numpy as np
import pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY='ca1d76dde41cf661ff6009df2c27b08f'

# Load model & scaler for current AQI prediction
model = pickle.load(open('aqi_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🌍 AQI Predictor | Current & 5-Hour Forecast")

mode = st.radio("Choose Mode", ["City", "Coordinates"])

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
        return response['list'][:6]  # now + next 5 hours
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
            st.success(f"📍 {city} (Lat: {lat}, Lon: {lon})")
            components = get_current_pollutants(lat, lon)
            forecast = get_forecast_aqi(lat, lon)
        else:
            st.error("City not found.")

elif mode == "Coordinates":
    lat = st.number_input("Latitude", format="%.4f")
    lon = st.number_input("Longitude", format="%.4f")
    if st.button("Get AQI"):
        location_name = f"({lat},{lon})"
        components = get_current_pollutants(lat, lon)
        forecast = get_forecast_aqi(lat, lon)

# AQI Display Section
if components:
    st.subheader("📊 Current Pollutants (µg/m³):")
    st.json(components)

    input_data = np.array([[components.get(k, 0) for k in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]])
    scaled_data = scaler.transform(input_data)
    current_prediction = model.predict(scaled_data)[0]

    current_time = datetime.now()
    st.header(f"AQI for {location_name}")
    st.success(f"✅ Now ({current_time.strftime('%H:%M:%S')}): {current_prediction:.2f}")

if forecast:
    st.subheader("🕒 Next 5-Hour AQI Forecast (via API):")
    for entry in forecast:
        timestamp = datetime.utcfromtimestamp(entry['dt']) + timedelta(hours=5.5)  # Adjust to IST
        aqi = entry['main']['aqi']  # API gives AQI as category 1-5
        pollutants = entry['components']
        st.write(f"**{timestamp.strftime('%H:%M')} IST** ➡️ AQI Category: {aqi}")
        with st.expander("Pollutants"):
            st.json(pollutants)
