from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load("asthma_model.pkl")

@app.route('/')
def home():
    return "Welcome to the Asthma Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get("symptoms")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        prediction = model.predict([symptoms])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Ensure port is 5000
