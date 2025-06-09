import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import folium
from streamlit_folium import folium_static
import requests

# --- Title and Description ---
st.title("🚦 AI-Powered Traffic Flow Optimization")
st.markdown("Upload traffic data, filter by location and time, predict congestion, and compare with real-time traffic.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your traffic dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # --- Sidebar Filters ---
    st.sidebar.header("📍 Filter Data")
    city_options = df['City'].unique().tolist()
    selected_city = st.sidebar.selectbox("Select City", city_options)
    selected_hour = st.sidebar.slider("Select Hour of Day", 0, 23, 12)
    day_options = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    selected_day = st.sidebar.selectbox("Select Day", options=list(day_options.keys()), format_func=lambda x: day_options[x])

    # --- Feature Engineering ---
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    le = LabelEncoder()
    df['Congestion_Label'] = le.fit_transform(df['Congestion_Level'])

    # --- Apply Filters ---
    filtered_df = df[
        (df['City'] == selected_city) &
        (df['Hour'] == selected_hour) &
        (df['DayOfWeek'] == selected_day)
    ]

    st.success(f"🔎 Showing data for: {selected_city} | Hour: {selected_hour} | Day: {day_options[selected_day]}")
    st.write(f"Total Records: {filtered_df.shape[0]}")

    if filtered_df.empty:
        st.warning("No data available for this selection.")
    else:
        # --- Modeling ---
        features = ['Hour', 'DayOfWeek', 'Is_Weekend', 'City', 'Road_Name', 'Direction', 'Lanes', 'Speed_kmph']
        X_filtered = filtered_df[features]
        y_filtered = filtered_df['Congestion_Label']

        cat_features = ['City', 'Road_Name', 'Direction']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough')

        rf_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        rf_model.fit(X_filtered, y_filtered)
        rf_preds = rf_model.predict(X_filtered)
        filtered_df['RF_Predicted'] = le.inverse_transform(rf_preds)

        xgb_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])
        xgb_model.fit(X_filtered, y_filtered)
        xgb_preds = xgb_model.predict(X_filtered)
        filtered_df['XGB_Predicted'] = le.inverse_transform(xgb_preds)

        # --- Show Predictions ---
        st.subheader("📊 Prediction Results")
        st.dataframe(filtered_df[['Timestamp', 'City', 'Road_Name', 'Direction', 'Speed_kmph', 'Congestion_Level', 'RF_Predicted', 'XGB_Predicted']].head())

        # --- Visuals ---
        st.subheader("🔍 Congestion Predictions")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(x=filtered_df['RF_Predicted'], order=le.classes_, palette='Set2', ax=ax1)
            ax1.set_title("Random Forest")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(x=filtered_df['XGB_Predicted'], order=le.classes_, palette='Set3', ax=ax2)
            ax2.set_title("XGBoost")
            st.pyplot(fig2)

        # --- Heatmap using folium (mock location) ---
        st.subheader("🗺️ Congestion Heatmap")
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)  # Default to Delhi center
        for _, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[28.6139 + 0.01 * (hash(row['Road_Name']) % 5), 77.2090 + 0.01 * (hash(row['Direction']) % 5)],
                radius=7,
                color='red' if row['RF_Predicted'] == 'High' else 'orange' if row['RF_Predicted'] == 'Moderate' else 'green',
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['Road_Name']} | {row['RF_Predicted']}"
            ).add_to(m)
        folium_static(m)

        # --- Real-time from HERE API ---
        st.subheader("📡 Real-Time Traffic (Mocked)")
        st.markdown("(This is a mockup. Replace with actual HERE API calls if API key is available.)")

        # Simulated HERE API output
        st.write("Live congestion on NH48 (Delhi):")
        st.info("🚗 Speed: 18.5 km/h | Congestion: High")

        # --- Download Result ---
        st.download_button(
            "📥 Download Enhanced Dataset as CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_traffic_predictions.csv",
            mime="text/csv"
        )


