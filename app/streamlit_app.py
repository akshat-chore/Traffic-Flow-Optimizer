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
from streamlit_folium import st_folium

st.set_page_config(page_title="AI Traffic Optimizer", layout="wide")
st.title("🚦 AI-Powered Traffic Flow Optimizer")

# --- Caching Optimized ---
@st.cache_data(ttl=600)
def load_data(file):
    return pd.read_csv(file)

# --- Sample Dataset Download ---
with st.expander("📁 Download a sample dataset to try the app"):
    try:
        sample_df = load_data("data/xmap_traffic_data.csv")
        st.download_button(
            "📥 Download Sample Dataset",
            data=sample_df.to_csv(index=False),
            file_name="sample_traffic_dataset.csv",
            mime="text/csv"
        )
    except Exception:
        st.error("🚫 Sample dataset not found. Ensure it exists at 'data/xmap_traffic_data.csv'.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your traffic dataset (.csv)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Traffic Data")
    city_options = df['City'].unique().tolist()
    selected_city = st.sidebar.selectbox("City", city_options)
    selected_hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    selected_day = st.sidebar.selectbox("Day of Week", list(day_map.keys()), format_func=lambda x: day_map[x])

    # --- Feature Engineering ---
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    le = LabelEncoder()
    df['Congestion_Label'] = le.fit_transform(df['Congestion_Level'])

    # --- Filtered Data ---
    filtered_df = df[
        (df['City'] == selected_city) &
        (df['Hour'] == selected_hour) &
        (df['DayOfWeek'] == selected_day)
    ]

    st.markdown("### 📊 Filtered Data")
    st.write(f"Records Found: {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(10))  # Limit rows for faster UI rendering

    # --- ML Predictions ---
    if filtered_df.shape[0] >= 3 and filtered_df['Congestion_Level'].nunique() > 1:
        features = ['Hour', 'DayOfWeek', 'Is_Weekend', 'City', 'Road_Name', 'Direction', 'Lanes', 'Speed_kmph']
        X = filtered_df[features]
        y = le.fit_transform(filtered_df['Congestion_Level'])

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['City', 'Road_Name', 'Direction'])
        ], remainder='passthrough')

        # --- Optimized RandomForest ---
        rf_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42))
        ])
        rf_model.fit(X, y)
        filtered_df['RF_Predicted'] = le.inverse_transform(rf_model.predict(X))

        # --- Optimized XGBoost ---
        xgb_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42))
        ])
        xgb_model.fit(X, y)
        filtered_df['XGB_Predicted'] = le.inverse_transform(xgb_model.predict(X))

        st.markdown("### 🤖 Congestion Predictions")
        st.dataframe(filtered_df[['Timestamp', 'City', 'Road_Name', 'Speed_kmph', 'Congestion_Level', 'RF_Predicted', 'XGB_Predicted']].head())

        # --- Visualizations ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Random Forest")
            fig1, ax1 = plt.subplots()
            sns.countplot(x=filtered_df['RF_Predicted'], order=le.classes_, palette='Set2', ax=ax1)
            ax1.set_title("RF Predicted Congestion")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### XGBoost")
            fig2, ax2 = plt.subplots()
            sns.countplot(x=filtered_df['XGB_Predicted'], order=le.classes_, palette='Set3', ax=ax2)
            ax2.set_title("XGB Predicted Congestion")
            st.pyplot(fig2)

        # --- Map View ---
        st.markdown("### 🗺️ Congestion Map View")
        city_coords = {
            "Delhi": [28.6139, 77.2090],
            "Bangalore": [12.9716, 77.5946],
            "Mumbai": [19.0760, 72.8777],
            "Chennai": [13.0827, 80.2707],
            "Hyderabad": [17.3850, 78.4867]
        }
        map_center = city_coords.get(selected_city, [20.5937, 78.9629])
        m = folium.Map(location=map_center, zoom_start=12)

        # Limit markers to prevent lag
        for _, row in filtered_df.head(50).iterrows():
            color = 'green' if row['XGB_Predicted'] == 'Low' else 'orange' if row['XGB_Predicted'] == 'Moderate' else 'red'
            folium.CircleMarker(
                location=map_center,
                radius=4,
                popup=f"{row['Road_Name']}: {row['XGB_Predicted']}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=450)

        # --- Download Option ---
        st.download_button(
            label="📥 Download Enhanced Dataset",
            data=filtered_df.to_csv(index=False),
            file_name="traffic_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ No predictions due to limited data.")

else:
    st.warning("⚠️ Please upload a dataset to get started.")