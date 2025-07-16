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
    r = requests.get(url).json()
    return (r[0]['lat'], r[0]['lon']) if r else (None, None)

def get_pollutants(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url).json()
    return r['list'][0]['components'] if 'list' in r else None

st.title("🌍 Air Quality Index Predictor")

mode = st.radio("Choose mode:", ("City", "Coordinates"))

if mode == "City":
    city = st.text_input("Enter City")
    if city:
        lat, lon = get_latlon(city)
        if lat:
            components = get_pollutants(lat, lon)
            if components:
                data = np.array([[components[k] for k in ['co','no','no2','o3','so2','pm2_5','pm10','nh3']]])
                pred = model.predict(scaler.transform(data))[0]
                st.success(f"AQI Prediction: {pred:.2f}")
                st.json(components)
            else:
                st.error("Couldn't fetch pollutant data!")
        else:
            st.error("City not found!")

elif mode == "Coordinates":
    lat = st.number_input("Latitude")
    lon = st.number_input("Longitude")
    if st.button("Predict AQI"):
        components = get_pollutants(lat, lon)
        if components:
            data = np.array([[components[k] for k in ['co','no','no2','o3','so2','pm2_5','pm10','nh3']]])
            pred = model.predict(scaler.transform(data))[0]
            st.success(f"AQI Prediction: {pred:.2f}")
            st.json(components)
        else:
            st.error("Couldn't fetch pollutant data!")
