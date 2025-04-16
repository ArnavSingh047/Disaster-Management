import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium

# Load historical flood data
data = pd.read_csv('./Dataset/Flood-Data.csv')

# Preprocessing: Assuming relevant features include Latitude, Longitude, Rainfall, and Elevation
data.columns = [
    "Latitude", "Longitude", "Rainfall_mm", "Temperature_C", "Humidity_pct", 
    "River_Discharge_m3_s", "Water_Level_m", "Elevation_m", "Land_Cover", 
    "Soil_Type", "Population_Density", "Infrastructure", "Historical_Floods", 
    "Flood_Occurred"
]

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_columns = ['Land_Cover', 'Soil_Type']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target variable
X = data[['Latitude', 'Longitude', 'Rainfall_mm', 'Elevation_m', 'River_Discharge_m3_s']]  # Add features based on dataset relevance
y = data['Flood_Occurred']  # Binary target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Load district data (with 'Lat', 'Long', and 'State' columns)
district_df = pd.read_csv('./Dataset/Flood-Predicting-Cities.csv')

# Rename 'Lat' and 'Long' to 'Latitude' and 'Longitude' to match training data
district_df.rename(columns={'Lat': 'Latitude', 'Long': 'Longitude'}, inplace=True)

# Assuming the Excel sheet already contains 'Rainfall_mm', 'Elevation_m', and 'River_Discharge_m3_s' columns for each city
# You can directly load the values from the dataset and apply predictions

# Predict flood risks for each district
district_df['Predicted_Flood_Risk'] = rf_model.predict(
    district_df[['Latitude', 'Longitude', 'Rainfall_mm', 'Elevation_m', 'River_Discharge_m3_s']]
)

# Classify districts based on predicted flood risk
def classify_flood_risk(risk_score):
    if risk_score < 0.5:
        return 'Low'
    elif 0.5 <= risk_score < 0.65:
        return 'Medium'
    else:
        return 'High'

district_df['Risk'] = district_df['Predicted_Flood_Risk'].apply(classify_flood_risk)

# Save the results to a new Excel file
district_df.to_excel('Predicted_Flood_Areas.xlsx', index=False)

print(district_df[['City', 'State', 'Predicted_Flood_Risk', 'Risk']])

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define color mapping for risk categories
color_map = {
    'Low': 'green',
    'Medium': 'orange',
    'High': 'red'
}

# Add markers for each district with the corresponding risk color
for index, row in district_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=7,
        popup=f"{row['City']} ({row['State']}): {row['Risk']} - Predicted Risk Score: {row['Predicted_Flood_Risk']:.2f}",
        color=color_map[row['Risk']],
        fill=True,
        fill_color=color_map[row['Risk']],
        fill_opacity=0.6
    ).add_to(india_map)

# Save the map to an HTML file
india_map.save('flood_risk_map.html')

print("Map has been saved as 'flood_risk_map.html'.")
