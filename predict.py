import streamlit as st
import requests
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import os
from pytz import timezone
import matplotlib.pyplot as plt
import json

# Load environment variables
load_dotenv()
API_KEY = 'ca1d76dde41cf661ff6009df2c27b08f'

# Load ML model and scaler
model = pickle.load(open('aqi_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Custom styles: white inputs, button, background
page_styles = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-photo/brown-cream-aesthetic-linktree-background-1_873036-161.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #1a1a1a !important;
}
h1, h2, h3, h4, h5, h6, p, span, div, label, button, input, .stRadio label {
    color: #1a1a1a !important;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
input, textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
}
.stNumberInput input, .stTextInput input {
    background-color: white !important;
    color: black !important;
}
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}
div.stButton > button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
    padding: 0.5em 1.2em;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #f0f0f0 !important;
}
.custom-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
    font-size: 17px;
    color: black;
    margin-bottom: 20px;
    overflow-x: auto;
}
code {
    background-color: #f3f3f3 !important;
    color: #222 !important;
}
</style>
'''
st.markdown(page_styles, unsafe_allow_html=True)

# App title and mode selection
st.title("🌍 AQI Predictor | Current + Next 5 Hours Forecast with ML")
mode = st.radio("📍 Choose Mode", ["City", "Coordinates"])

# Timezone
ist = timezone('Asia/Kolkata')

# Utility Functions
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

# Inputs
components = None
forecast = None
location_name = ""

if mode == "City":
    city = st.text_input("🏙️ Enter City Name")
    if st.button("Get AQI"):
        lat, lon = get_latlon(city)
        if lat:
            location_name = city
            st.success(f"📌 {city} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            components = get_current_pollutants(lat, lon)
            forecast = get_forecast_aqi(lat, lon)
        else:
            st.error("❌ City not found. Please check spelling or try a nearby location.")

elif mode == "Coordinates":
    lat = st.number_input("Latitude", format="%.4f")
    lon = st.number_input("Longitude", format="%.4f")
    if st.button("Get AQI"):
        location_name = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        components = get_current_pollutants(lat, lon)
        forecast = get_forecast_aqi(lat, lon)

# Current AQI Output
if components:
    st.markdown('<div class="custom-box"><b>📊 Current Pollutants (µg/m³)</b><br><pre>' + 
                json.dumps(components, indent=2) + '</pre></div>', unsafe_allow_html=True)

    input_data = np.array([[components.get(k, 0) for k in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]])
    scaled_data = scaler.transform(input_data)
    current_prediction = model.predict(scaled_data)[0]

    current_time = datetime.now(ist)
    st.markdown(f'''
    <div class="custom-box">
        ✅ <b>AQI for {location_name}</b><br><br>
        <b>Now ({current_time.strftime("%H:%M:%S")} IST)</b><br>
        ML Predicted AQI = <b>{current_prediction:.2f}</b>
    </div>''', unsafe_allow_html=True)

# Forecast Section
if forecast:
    st.subheader("🕒 Next 5 Hours AQI Forecast")

    current_timestamp = datetime.now().timestamp()
    upcoming_forecasts = [entry for entry in forecast if entry['dt'] > current_timestamp][:5]

    if not upcoming_forecasts:
        st.warning("⚠️ No forecast data available for the next 5 hours.")
    else:
        forecast_times = []
        forecast_aqi = []

        for entry in upcoming_forecasts:
            timestamp = datetime.fromtimestamp(entry['dt'], ist)
            pollutants = entry['components']
            forecast_times.append(timestamp.strftime("%H:%M"))

            future_data = np.array([[pollutants.get(k, 0) for k in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]])
            future_scaled = scaler.transform(future_data)
            future_aqi_pred = model.predict(future_scaled)[0]

            forecast_aqi.append(future_aqi_pred)

            st.markdown(f'''
            <div class="custom-box">
                ⏰ <b>{timestamp.strftime("%H:%M")} IST</b><br>
                Predicted AQI = <b>{future_aqi_pred:.2f}</b><br><br>
                <b>Pollutants:</b><br><pre>{json.dumps(pollutants, indent=2)}</pre>
            </div>
            ''', unsafe_allow_html=True)

        # Visualization
        st.subheader("📈 Forecasted AQI Line Chart")
        fig, ax = plt.subplots()
        ax.plot(forecast_times, forecast_aqi, marker='o', color='skyblue', linewidth=2)
        ax.set_xlabel("Time (IST)")
        ax.set_ylabel("Predicted AQI")
        ax.set_title(f"AQI Forecast for {location_name}")
        ax.grid(True)
        st.pyplot(fig)
