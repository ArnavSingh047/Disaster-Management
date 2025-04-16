import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib  # For saving the trained model

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
from sklearn.preprocessing import OneHotEncoder

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

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Save the trained model to a file
joblib.dump(rf_model, 'forest_fire_model.pkl')
print("Forest fire prediction model has been saved as 'forest_fire_model.pkl'.")
