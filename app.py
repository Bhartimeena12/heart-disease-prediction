import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model files
# -----------------------------

model = joblib.load("Logistic Regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("❤️ Heart Disease Prediction")

st.write("Please provide the following details.")


age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak",
    min_value=0.0,
    max_value=6.0,
    value=1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    st.write("Creating input...")

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    st.write("Input created successfully")

    # Missing columns ko 0 se fill karna
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Same column order as training
    input_df = input_df[expected_columns]

    st.write("Columns matched successfully")

    # Scaling
    scaled_input = scaler.transform(input_df)

    st.write("Scaling successful")

    # Prediction
    prediction = model.predict(scaled_input)[0]

    st.write("Prediction completed")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")