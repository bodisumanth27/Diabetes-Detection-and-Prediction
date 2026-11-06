import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
MODEL_PATH = "diabetes_model.pkl"

st.title("🩺 Diabetes Detection and Prediction")
st.write(
    "An educational machine-learning application "
    "for diabetes risk classification."
)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# Input form
st.subheader("Enter Health Measurements")

with st.form("prediction_form"):

    pregnancies = st.number_input(
        "Number of pregnancies",
        min_value=0,
        max_value=30,
        value=1
    )

    glucose = st.number_input(
        "Glucose concentration (mg/dL)",
        min_value=1.0,
        max_value=500.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Diastolic blood pressure (mmHg)",
        min_value=1.0,
        max_value=250.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin thickness (mm)",
        min_value=0.0,
        max_value=150.0,
        value=20.0
    )

    insulin = st.number_input(
        "Insulin (mu U/mL)",
        min_value=0.0,
        max_value=3000.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )

    pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=5.0,
        value=0.5
    )

    age = st.number_input(
        "Age (years)",
        min_value=21,
        max_value=120,
        value=30
    )

    submitted = st.form_submit_button(
        "Predict"
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    st.metric(
        "Model-estimated probability for class 1",
        f"{probabilities[1] * 100:.1f}%"
    )

    if prediction == 1:
        st.warning(
            "The model classified this input as "
            "the diabetes-positive class."
        )
    else:
        st.success(
            "The model classified this input as "
            "the diabetes-negative class."
        )