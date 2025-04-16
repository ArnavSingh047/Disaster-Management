import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import folium

# Load historical forest fire data
data = pd.read_csv('./Dataset/ForestFire-Data.csv')

# Clean column names to avoid any inconsistencies
data.columns = data.columns.str.strip()

# Preprocessing: Calculate total occurrences for each state based on historical data
data['Total_Occurrences'] = data[[ 
    'January 2018 to June 2018', 
    'November 2018 to June 2019',
    'November 2019 to June 2020',
    'November 2020 to June 2021',
    'November 2021 to June 2022',
    'November 2022 to June 2023'
]].sum(axis=1)

# Classify risk based on total occurrences
def classify_risk(total_occurrences):
    if total_occurrences <= 5000:
        return 0  # Less Prone
    elif 5001 <= total_occurrences <= 12000:
        return 1  # Medium Prone
    else:
        return 2  # Highly Prone

data['Risk_Level'] = data['Total_Occurrences'].apply(classify_risk)

# Group data by State/UT and calculate average total occurrences for each state
state_occurrences = data.groupby('State/UT')['Total_Occurrences'].mean()  # Average total occurrences per state
state_occurrences_df = pd.DataFrame(state_occurrences).reset_index()

# Now, prepare features for training (use state names as features)
X = state_occurrences_df[['State/UT']]  # State names as features
y = state_occurrences_df['Total_Occurrences'].apply(classify_risk)  # Target variable: Risk level

# Use OneHotEncoder to handle state names as categorical features
encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Load cities data for prediction
cities_df = pd.read_csv('./Dataset/ForestFire-Prediction-Cities.csv')

# Map the state occurrences to the cities based on their state
cities_df['State_Occurrences'] = cities_df['State/UT'].map(state_occurrences)

# One-hot encode the states in cities_df, using the same encoder as during training
X_cities_encoded = encoder.transform(cities_df[['State/UT']])

# Predict risk levels for cities based on their state occurrences
cities_df['Predicted_Risk_Level'] = rf_model.predict(X_cities_encoded)

# Map numerical risk levels to textual categories
risk_mapping = {0: 'Less Prone', 1: 'Medium Prone', 2: 'Highly Prone'}
cities_df['Risk_Category'] = cities_df['Predicted_Risk_Level'].map(risk_mapping)

# Save predictions to a new Excel file
cities_df.to_excel('Predicted-ForestFire-Areas.xlsx', index=False)

print("Predictions saved to 'Predicted-ForestFire-Areas.xlsx'.")

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define color mapping for risk categories
color_map = {
    'Less Prone': 'green',
    'Medium Prone': 'orange',
    'Highly Prone': 'red'
}

# Add markers for each city with the corresponding risk color
for index, row in cities_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],  # Adjusted based on correct column names
        radius=7,
        popup=f"{row['State/UT']}: {row['Risk_Category']}",
        color=color_map[row['Risk_Category']],
        fill=True,
        fill_color=color_map[row['Risk_Category']],
        fill_opacity=0.6
    ).add_to(india_map)

# Save the map to an HTML file
india_map.save('forest_fire_risk_map.html')

print("Map has been saved as 'forest_fire_risk_map.html'.")
