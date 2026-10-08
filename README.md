# Student Performance Predictor

A simple machine-learning project that predicts whether a student is likely to pass based on study hours, attendance, previous score, and assignment completion.

## ML approach
- Algorithm: Random Forest Classifier
- Train/test split: 75% / 25%
- Test accuracy on the included sample dataset: 100.0%

## Technologies
Python, Pandas, Scikit-learn, Joblib, Streamlit.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files
- `app.py` — interactive prediction app
- `student_performance.csv` — sample dataset
- `model.joblib` — trained ML model
- `requirements.txt` — dependencies
