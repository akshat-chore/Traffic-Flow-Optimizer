import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/xmap_traffic_data.csv')

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Preview
print("\n📊 First 5 Rows:")
print(df.head())

print("\n🧾 Columns:")
print(df.columns)

print("\n📦 Data Types:")
print(df.dtypes)

print("\n❓ Missing Values:")
print(df.isnull().sum())

# --- Basic Visualizations ---

# 1. Congestion level distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Congestion_Level', order=['Low', 'Moderate', 'High'], palette='coolwarm')
plt.title("Traffic Congestion Level Distribution")
plt.tight_layout()
plt.show()

# 2. Speed distribution by congestion level
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Congestion_Level', y='Speed_kmph', order=['Low', 'Moderate', 'High'], palette='Set2')
plt.title("Speed Distribution by Congestion Level")
plt.tight_layout()
plt.show()

# 3. Average speed by hour
df['Hour'] = df['Timestamp'].dt.hour
hourly_speed = df.groupby('Hour')['Speed_kmph'].mean().reset_index()

plt.figure(figsize=(8,4))
sns.lineplot(data=hourly_speed, x='Hour', y='Speed_kmph', marker='o')
plt.title("Average Speed by Hour of Day")
plt.tight_layout()
plt.show()
