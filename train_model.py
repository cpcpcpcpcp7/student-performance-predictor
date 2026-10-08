import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("student_performance.csv")
X = df.drop(columns="pass")
y = df["pass"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "model.joblib")
print("Model trained and saved as model.joblib")
