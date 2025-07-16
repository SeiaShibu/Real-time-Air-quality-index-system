# 🌍 Real-time Air Quality Index (AQI) Predictor

This project predicts the **Air Quality Index (AQI)** for any location using real-time pollutant data from OpenWeather API and a machine learning model trained on historical AQI data.

### 🚀 Features
- 🌐 Fetch real-time pollutant levels using city name or latitude & longitude.
- 🧠 Predict AQI using a trained Random Forest Regressor model.
- 📊 Display pollutant levels alongside AQI predictions.
- 🌟 Beautiful and interactive Streamlit web interface.
- 🔒 Secure API key management using `.env`.

---

## 📁 Project Structure
├── data/
│ └── aqi_data_fetched1.csv # Dataset used for training
├── aqi_model.pkl # Trained machine learning model
├── scaler.pkl # Scaler for input data
├── train.py # Model training script
├── app.py # Streamlit app for AQI prediction
├── .env # API key (not uploaded to GitHub)
├── requirements.txt # Python dependencies
└── README.md # Project description

yaml
Copy
Edit

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/Real-time-Air-quality-index-system.git
cd Real-time-Air-quality-index-system
2. Setup Environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
3. Add OpenWeather API Key
Create a .env file in the root folder and paste:

ini
Copy
Edit
API_KEY=your_openweather_api_key
4. Train the Model (Optional)
bash
Copy
Edit
python train.py
5. Run Streamlit App
bash
Copy
Edit
streamlit run app.py
🌐 Live Demo
Access the app live here 👉"https://real-time-air-quality-index-system-jszvhulvpkhdbzdtfp5kdh.streamlit.app/"
📌 Tech Stack
Python (Pandas, Scikit-learn, Requests)

Streamlit

OpenWeather API

GitHub + Streamlit Cloud Deployment

🙏 Acknowledgements
OpenWeather for free air quality data.

Streamlit for providing an easy deployment solution.
