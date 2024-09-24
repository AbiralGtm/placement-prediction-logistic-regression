from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib  # For saving/loading models

app = Flask(__name__)

# Load the trained model (you can save it using joblib)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML page to input CGPA and IQ

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    
    # Make prediction using the model
    prediction = model.predict([[cgpa, iq]])[0]
    
    return jsonify({'placement': 'Placed' if prediction == 1 else 'Not Placed'})

if __name__ == '__main__':
    app.run(debug=True)
