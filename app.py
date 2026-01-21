from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and selected features
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')
selected_features = joblib.load('model/selected_features.pkl')

# Define cultivar names
cultivar_names = {
    0: 'Cultivar 0',
    1: 'Cultivar 1',
    2: 'Cultivar 2'
}


@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html', features=selected_features)


@app.route('/predict', methods=['POST'])
def predict():
    """Process the form data and make a prediction."""
    try:
        # Get the input values from the form
        input_values = []
        for feature in selected_features:
            value = float(request.form[feature])
            input_values.append(value)

        # Convert to numpy array and reshape for the model
        input_array = np.array(input_values).reshape(1, -1)

        # Scale the input data
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Get the cultivar name
        predicted_cultivar = cultivar_names[prediction]

        # Create a dictionary with the probabilities for each class
        probabilities = {
            cultivar_names[i]: f"{prob:.4f}" for i, prob in enumerate(prediction_proba)
        }

        # Render the result page
        return render_template('result.html',
                               prediction=predicted_cultivar,
                               probabilities=probabilities,
                               features=selected_features,
                               input_values=input_values)

    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)