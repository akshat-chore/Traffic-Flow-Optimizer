import pandas as pd
import random
from datetime import datetime, timedelta

# Define locations and roads
cities = ["Delhi", "Bangalore", "Mumbai", "Chennai", "Hyderabad"]
roads = {
    "Delhi": ["Ring Road", "Outer Ring Road", "NH48"],
    "Bangalore": ["100 Feet Rd", "Outer Ring Rd", "Bellary Rd"],
    "Mumbai": ["Western Express Hwy", "Eastern Freeway", "Linking Rd"],
    "Chennai": ["Mount Rd", "OMR", "ECR"],
    "Hyderabad": ["Necklace Rd", "Hitech City Rd", "NH65"]
}
directions = ["Northbound", "Southbound", "Eastbound", "Westbound"]
congestion_levels = ["Low", "Moderate", "High"]

# Generate traffic data
def generate_traffic_data(n=3000):
    data = []
    base_time = datetime(2024, 1, 1, 0, 0)

    for _ in range(n):
        city = random.choice(cities)
        road = random.choice(roads[city])
        direction = random.choice(directions)
        lanes = random.choice([1, 2, 3, 4])
        timestamp = base_time + timedelta(minutes=random.randint(0, 30*24*60))
        hour = timestamp.hour

        # Simulate congestion
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            congestion = random.choices(["High", "Moderate", "Low"], weights=[0.6, 0.3, 0.1])[0]
        else:
            congestion = random.choices(["Low", "Moderate", "High"], weights=[0.5, 0.3, 0.2])[0]

        # Simulate speed based on congestion
        speed = round(random.uniform(10, 25), 1) if congestion == "High" else \
                round(random.uniform(25, 40), 1) if congestion == "Moderate" else \
                round(random.uniform(40, 60), 1)

        data.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "City": city,
            "Road_Name": road,
            "Direction": direction,
            "Lanes": lanes,
            "Speed_kmph": speed,
            "Congestion_Level": congestion
        })

    return pd.DataFrame(data)

# Create DataFrame
df = generate_traffic_data(3000)

# Save to CSV
df.to_csv("data/xmap_traffic_data.csv", index=False)

print("✅ Synthetic traffic data saved to data/xmap_traffic_data.csv")
print(df.head())
