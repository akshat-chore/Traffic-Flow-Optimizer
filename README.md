# 🚦 Traffic Analyzer

An AI-powered ML project that analyzes and predicts traffic congestion using Random Forest and XGBoost. It leverages historical and real-time data to help optimize traffic flow and improve urban mobility.

---

## 📊 Features

- Predicts congestion levels using trained ML models.
- Built with Streamlit for an interactive dashboard.
- Filtering by city, time, and day of the week.
- Visualizes congestion patterns and flow heatmaps.

---

Models Used
Random Forest Regressor
XGBoost Regressor


Traffic-Analyzer/
│
├── app/
│   └── streamlit_app.py         # Main Streamlit application
│
├── data/
│   ├── traffic_data.csv         # Historical traffic dataset
│
├── models/
│   ├── rf_model.pkl             # Trained Random Forest model
│   └── xgb_model.pkl            # Trained XGBoost model
│
├── requirements.txt
└── README.md

## 🔧 How to Run the App

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Traffic-Analyzer.git


2. Navigate to the project directory:

cd Traffic-Analyzer

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the Streamlit app:

streamlit run app/streamlit_app.py

