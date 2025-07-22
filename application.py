from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = "artifacts/model/model.pkl"
scaler_path = "artifacts/processed/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template("index.html", predictions = None)

@app.route('/predict',methods=["POST"])
def predict():
    try:
        Healthcare_Cost = float(request.form["healthcare_costs"])
        Tumor_size = float(request.form["tumor_size"])
        Treatment_type = int(request.form["treatment_type"])
        Diabetes = int(request.form["diabetes"])
        Mortality_rate = float(request.form["mortality_rate"])

        input = np.array([[Healthcare_Cost,Tumor_size,Treatment_type,Diabetes,Mortality_rate]])

        scaled_input = scaler.transform(input)

        prediction = model.predict(scaled_input)[0]

        return render_template('index.html', prediction=prediction)
    
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True, host = "0.0.0.0", port = 5000)
    


