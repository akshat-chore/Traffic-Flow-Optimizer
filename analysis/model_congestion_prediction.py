from xgboost import XGBClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data/xmap_traffic_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# --- Feature Engineering ---
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = Monday
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# --- Target Variable ---
y = df['Congestion_Level']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Congestion_Label'] = le.fit_transform(df['Congestion_Level'])


# --- Feature Set ---
X = df[['Hour', 'DayOfWeek', 'Is_Weekend', 'City', 'Road_Name', 'Direction', 'Lanes', 'Speed_kmph']]

# --- One-Hot Encoding for Categorical Features ---
categorical_features = ['City', 'Road_Name', 'Direction']
numeric_features = ['Hour', 'DayOfWeek', 'Is_Weekend', 'Lanes', 'Speed_kmph']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# --- ML Pipeline ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_xgb, y_test_xgb = train_test_split(X, df['Congestion_Label'], test_size=0.2, random_state=42)

# --- Train the Model ---
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Visualize prediction results
sns.countplot(x=y_pred, order=['Low', 'Moderate', 'High'], palette='Set2')
plt.title("Predicted Congestion Levels")
plt.tight_layout()
plt.show()
# --- XGBoost Model ---
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

print("\n🚀 Training XGBoost Classifier...")
xgb_model.fit(X_train, y_train_xgb)
y_xgb_pred = xgb_model.predict(X_test)

# Convert XGBoost numeric predictions back to original labels
decoded_preds = le.inverse_transform(y_xgb_pred)
decoded_true = le.inverse_transform(y_test_xgb)

print("\n🎯 XGBoost Accuracy:", accuracy_score(decoded_true, decoded_preds))
print("\n📊 XGBoost Confusion Matrix:\n", confusion_matrix(decoded_true, decoded_preds))
print("\n📋 XGBoost Classification Report:\n", classification_report(decoded_true, decoded_preds))

# Visualize XGBoost predictions
sns.countplot(x=decoded_preds, order=['Low', 'Moderate', 'High'], palette='Set3')
plt.title("XGBoost Predicted Congestion Levels")
plt.tight_layout()
plt.show()
