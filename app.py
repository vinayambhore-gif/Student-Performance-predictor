from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")


# Home page (frontend UI)
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route used by the HTML form
@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["StudyHours"])
    attendance = float(request.form["Attendance"])
    previous_marks = float(request.form["PreviousMarks"])
    assignments = float(request.form["Assignments"])

    features = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignments
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


# JSON API route (for Postman / programmatic use)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    features = np.array([[
        data["StudyHours"],
        data["Attendance"],
        data["PreviousMarks"],
        data["Assignments"]
    ]])
    prediction = model.predict(features)
    return jsonify({
        "Predicted Marks": round(prediction[0], 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
