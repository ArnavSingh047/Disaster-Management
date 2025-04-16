import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

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

# Convert predicted float values to binary (0 or 1) for accuracy calculation
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")
