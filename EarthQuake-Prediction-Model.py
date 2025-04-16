import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium

# Load historical earthquake data
data = pd.read_csv('./Dataset/EarthQuake-Data.csv')

# Preprocessing: Assuming 'Latitude', 'Longitude', 'Depth', 'Magnitude' are columns in the dataset
X = data[['Latitude', 'Longitude', 'Depth']]  # Features used for training
y = data['Magnitude']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Load district data (with 'Lat', 'Long', and 'State' columns)
district_df = pd.read_csv('./Dataset/EarthQuake-Prediction-Cities.csv')

# Rename 'Lat' and 'Long' to 'Latitude' and 'Longitude' to match training data
district_df.rename(columns={'Lat': 'Latitude', 'Long': 'Longitude'}, inplace=True)

# Add a placeholder for 'Depth' (use median or mean from historical data)
depth_placeholder = np.median(data['Depth'])  # You can use mean() if you prefer
district_df['Depth'] = depth_placeholder  # Add depth to the district dataframe

# Predict magnitudes for each district based on Lat, Long, and Depth
district_df['Predicted_Magnitude'] = rf_model.predict(district_df[['Latitude', 'Longitude', 'Depth']])

# Classify districts based on predicted magnitudes
def classify_risk(magnitude):
    if magnitude < 3.8:
        return 'Low'
    elif 3.8 <= magnitude < 4.3:
        return 'Medium'
    else:
        return 'High'

district_df['Risk'] = district_df['Predicted_Magnitude'].apply(classify_risk)

# Save the results to a new Excel file
district_df.to_excel('Predicted_Earthquake_Areas.xlsx', index=False)

print(district_df[['City', 'State', 'Predicted_Magnitude', 'Risk']])

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
        popup=f"{row['City']} ({row['State']}): {row['Risk']} - Magnitude: {row['Predicted_Magnitude']:.2f}",
        color=color_map[row['Risk']],
        fill=True,
        fill_color=color_map[row['Risk']],
        fill_opacity=0.6
    ).add_to(india_map)

# Save the map to an HTML file
india_map.save('earthquake_risk_map.html')

print("Map has been saved as 'earthquake_risk_map.html'.")
