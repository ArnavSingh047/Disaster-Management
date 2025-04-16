import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
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

# Generate Confusion Matrix Heatmap (using binned magnitude values for classification)
bins = [0, 3.8, 4.3, np.inf]
labels = ['Low', 'Medium', 'High']
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)

cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.savefig('confusion_matrix_heatmap.png')
plt.show()

# Generate Feature Importance Bar Chart
feature_importances = rf_model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('feature_importance.png')
plt.show()

# Generate Data Distribution Chart
sns.kdeplot(data['Magnitude'], label='Original Magnitude', shade=True, color='blue')
sns.kdeplot(y, label='Processed Magnitude', shade=True, color='orange')
plt.title('Data Distribution: Original vs. Processed')
plt.xlabel('Magnitude')
plt.ylabel('Density')
plt.legend()
plt.savefig('data_distribution.png')
plt.show()

# Generate Comparative Model Performance (Add dummy data for multiple models)
models = ['Random Forest']
accuracy = [1 - mse]  # Example using 1-MSE as a proxy for accuracy (replace with actual metric)
performance_data = pd.DataFrame({'Model': models, 'Accuracy': accuracy})
performance_data.set_index('Model').plot(kind='bar', figsize=(10, 6), colormap='coolwarm')
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.savefig('model_performance_comparison.png')
plt.show()

# Generate Risk Classification Map
district_df = pd.read_csv('./Dataset/EarthQuake-Prediction-Cities.csv')
district_df.rename(columns={'Lat': 'Latitude', 'Long': 'Longitude'}, inplace=True)
depth_placeholder = np.median(data['Depth'])  # Use median for depth
district_df['Depth'] = depth_placeholder
district_df['Predicted_Magnitude'] = rf_model.predict(district_df[['Latitude', 'Longitude', 'Depth']])
district_df['Risk'] = district_df['Predicted_Magnitude'].apply(lambda x: 'Low' if x < 3.8 else ('Medium' if x < 4.3 else 'High'))

