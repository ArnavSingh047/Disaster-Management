import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load historical earthquake data
data = pd.read_csv('./Dataset/EarthQuake-Data.csv')

# Preprocessing: Use 'Latitude', 'Longitude', 'Depth' as features
X = data[['Latitude', 'Longitude', 'Depth']]
y = data['Magnitude']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model for Flask
joblib.dump(rf_model, 'earthquake_model.pkl')
print("Model saved as 'earthquake_model.pkl'.")
