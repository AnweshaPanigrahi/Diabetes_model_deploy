from flask import Flask, request, jsonify, render_template
import joblib  # To load the model
import numpy as np

# Load the pre-trained model
model = joblib.load('diabetes_model.pkl')  # Updated model file name

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting values from form fields (order matters as per your dataset)
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = int(request.form['blood_glucose_level'])

        # Encoding categorical variables manually (example: gender and smoking_history)
        gender_encoded = 1 if gender == 'Male' else 0
        smoking_encoded = 1 if smoking_history == 'current' else 0  # Adjust based on your encoding logic

        # Prepare the input data for prediction
        input_features = np.array([gender_encoded, age, hypertension, heart_disease, 
                                   smoking_encoded, bmi, HbA1c_level, blood_glucose_level]).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_features)[0]

        # Map prediction to readable output
        result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"

        # Return the result to the webpage
        return render_template('index.html', prediction_text=f'Diabetes Prediction: {result}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Set debug=True for easier debugging, and specify a custom port if needed
    app.run(debug=True, port=8000)  # Change 'port=8000' if you want another port
