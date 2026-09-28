import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("data/student_data.csv")

X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignments"]]
y = data["FinalMarks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained successfully")
