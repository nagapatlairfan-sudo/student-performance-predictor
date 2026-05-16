from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model/student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])

    features = np.array([[hours, attendance, previous_marks]])

    prediction = model.predict(features)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)