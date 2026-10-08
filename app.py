import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student is likely to pass using basic academic indicators.")

model = joblib.load("model.joblib")

study = st.slider("Daily study hours", 0.0, 12.0, 5.0, 0.5)
attendance = st.slider("Attendance (%)", 0, 100, 80)
previous = st.slider("Previous score (%)", 0, 100, 65)
assignments = st.slider("Assignments completed", 0, 10, 7)

if st.button("Predict"):
    X = pd.DataFrame([[study, attendance, previous, assignments]],
                     columns=["study_hours","attendance","previous_score","assignments_completed"])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    if pred:
        st.success(f"Likely to PASS — probability: {prob:.1%}")
    else:
        st.error(f"At risk of FAILING — pass probability: {prob:.1%}")
