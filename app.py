from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model and label encoder
model = joblib.load('medication_model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return "Lung Cancer Medication Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract vitals from request
        heart_rate = float(data['Heart Rate (bpm)'])
        oxygen = float(data['Oxygen Saturation (%)'])
        temperature = float(data['Temperature (°C)'])

        # Prepare input
        input_df = pd.DataFrame([{
            'Heart Rate (bpm)': heart_rate,
            'Oxygen Saturation (%)': oxygen,
            'Temperature (°C)': temperature
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        medication = le.inverse_transform([prediction])[0]

        return jsonify({'prediction': medication})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
