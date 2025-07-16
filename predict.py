import streamlit as st
import requests, pickle, numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

model = pickle.load(open('aqi_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def get_latlon(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    response = requests.get(url).json()
    if response:
        return response[0]['lat'], response[0]['lon']
    return None, None

def get_pollutants(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url).json()
    if 'list' in response:
        return response['list'][0]['components']
    return None

st.title("🌍 Real-Time Air Quality Index Predictor")

mode = st.radio("Choose Mode", ["City", "Coordinates"])

if mode == "City":
    city = st.text_input("Enter City Name")
    if city:
        lat, lon = get_latlon(city)
        if lat:
            st.info(f"Latitude: {lat}, Longitude: {lon}")
            components = get_pollutants(lat, lon)
            if components:
                st.subheader("Pollutants Levels (µg/m³):")
                st.json(components)

                input_data = np.array([[components[k] for k in ['co','no','no2','o3','so2','pm2_5','pm10','nh3']]])
                scaled_data = scaler.transform(input_data)
                prediction = model.predict(scaled_data)[0]
                st.success(f"Predicted AQI for {city}: {prediction:.2f}")
            else:
                st.error("Couldn't fetch pollutant data.")
        else:
            st.error("City not found.")

if mode == "Coordinates":
    lat = st.number_input("Latitude", format="%.4f")
    lon = st.number_input("Longitude", format="%.4f")
    if st.button("Predict AQI"):
        components = get_pollutants(lat, lon)
        if components:
            st.subheader("Pollutants Levels (µg/m³):")
            st.json(components)

            input_data = np.array([[components[k] for k in ['co','no','no2','o3','so2','pm2_5','pm10','nh3']]])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            st.success(f"Predicted AQI: {prediction:.2f}")
        else:
            st.error("Couldn't fetch pollutant data.")
