from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, label encoder, and scaler
with open('fish_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)
    species = le.inverse_transform(prediction)[0]
    return jsonify({'prediction': species})

if __name__ == '__main__':
    app.run(debug=True)
