from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for React frontend

# Load the saved models
earthquake_model = joblib.load('D:/Mini Project/Web-Project/Backend/earthquake_model.pkl')
flood_model = joblib.load('D:/Mini Project/Web-Project/Backend/flood_model.pkl')
forest_fire_model = joblib.load('D:/Mini Project/Web-Project/Backend/forest_fire_model.pkl')

# Load the One-Hot Encoder for the Forest Fire model
forest_fire_encoder = joblib.load('D:/Mini Project/Web-Project/Backend/one_hot_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict earthquake magnitude, flood risk, or forest fire risk based on input data.
    """
    try:
        # Get JSON data from the request
        data = request.json
        model_type = data.get('model_type')  # 'earthquake', 'flood', or 'forestfire'

        if model_type == 'earthquake':
            # Get earthquake-related inputs
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            depth = data.get('depth')

            # Validate input
            if latitude is None or longitude is None or depth is None:
                return jsonify({'error': 'Missing input data for earthquake model. Provide latitude, longitude, and depth.'}), 400

            # Prepare input for the earthquake model
            features = [[latitude, longitude, depth]]

            # Make prediction
            prediction = earthquake_model.predict(features)
            return jsonify({'model_type': 'earthquake', 'predicted_magnitude': prediction[0]})

        elif model_type == 'flood':
            # Get flood-related inputs
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            rainfall_mm = data.get('rainfall_mm')
            elevation_m = data.get('elevation_m')
            river_discharge = data.get('river_discharge_m3_s')

            # Validate input
            if latitude is None or longitude is None or rainfall_mm is None or elevation_m is None or river_discharge is None:
                return jsonify({'error': 'Missing input data for flood model. Provide latitude, longitude, rainfall_mm, elevation_m, and river_discharge_m3_s.'}), 400

            # Prepare input for the flood model
            features = [[latitude, longitude, rainfall_mm, elevation_m, river_discharge]]

            # Make prediction
            prediction = flood_model.predict(features)
            return jsonify({'model_type': 'flood', 'predicted_flood_risk': prediction[0]})

        elif model_type == 'forestfire':
            # Get forest fire-related inputs
            state = data.get('state')  # Assuming you're predicting based on state for forest fire

            # Validate input
            if state is None:
                return jsonify({'error': 'Missing input data for forest fire model. Provide state.'}), 400

            # Prepare input for the forest fire model with One-Hot Encoding
            state_data = pd.DataFrame({'State/UT': [state]})
            encoded_state = forest_fire_encoder.transform(state_data)

            # Make prediction
            prediction = forest_fire_model.predict(encoded_state)

            # Map the numeric prediction to a risk level (for example)
            risk_levels = ['Low', 'Medium', 'High']
            predicted_risk = risk_levels[int(prediction[0])]

            return jsonify({'model_type': 'forestfire', 'predicted_risk_level': predicted_risk})

        else:
            return jsonify({'error': 'Invalid model_type. Must be "earthquake", "flood", or "forestfire".'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
