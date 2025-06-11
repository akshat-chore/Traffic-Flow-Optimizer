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
st.markdown("Make informed traffic decisions using machine learning models and visual insights.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your traffic dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
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
    st.dataframe(filtered_df.head())

    # --- Condition Checks ---
    if filtered_df.shape[0] < 3:
        st.warning("⚠️ Very few records — predictions may be less reliable.")
        show_predictions = True
    elif filtered_df['Congestion_Level'].nunique() < 2:
        st.warning("⚠️ Only one class present. Models need variety to train.")
        show_predictions = False
    else:
        show_predictions = True

    # --- ML and Visualizations ---
    if show_predictions:
        features = ['Hour', 'DayOfWeek', 'Is_Weekend', 'City', 'Road_Name', 'Direction', 'Lanes', 'Speed_kmph']
        X = filtered_df[features]
        le_y = LabelEncoder()
        y = le_y.fit_transform(filtered_df['Congestion_Level'])

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['City', 'Road_Name', 'Direction'])
        ], remainder='passthrough')

        rf_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        rf_model.fit(X, y)
        rf_preds = rf_model.predict(X)
        filtered_df['RF_Predicted'] = le.inverse_transform(rf_preds)

        xgb_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)
        filtered_df['XGB_Predicted'] = le_y.inverse_transform(xgb_preds)

        st.markdown("### 🤖 Congestion Predictions")
        st.dataframe(filtered_df[['Timestamp', 'City', 'Road_Name', 'Speed_kmph', 'Congestion_Level', 'RF_Predicted', 'XGB_Predicted']].head())

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
        for _, row in filtered_df.iterrows():
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

        # --- Real-Time Section (Mocked with summary) ---
        st.markdown("### 📡 Real-Time Traffic (Mocked)")
        st.info("🚧 Live HERE API integration will be added in future versions.")
        st.success(f"Current prediction summary: "
                   f"{(filtered_df['XGB_Predicted'] == 'Low').sum()} Low, "
                   f"{(filtered_df['XGB_Predicted'] == 'Moderate').sum()} Moderate, "
                   f"{(filtered_df['XGB_Predicted'] == 'High').sum()} High")

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
    with st.expander("📁 Or download a sample dataset"):
        try:
            real_data = pd.read_csv("data/xmap_traffic_data.csv")
            st.download_button(
                "📥 Download Real Sample Dataset",
                data=real_data.to_csv(index=False),
                file_name="real_traffic_dataset.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error("🚫 Could not load sample dataset from /data/ folder.")
