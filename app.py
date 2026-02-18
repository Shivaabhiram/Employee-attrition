import streamlit as st
import pandas as pd
import joblib

model = joblib.load("employee_attrition_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the company")

age = st.number_input("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", 1000, 200000)
years_at_company = st.number_input("Years at Company", 0, 40)
job_satisfaction = st.slider("Job Satisfaction (1 = low, 4 = High)", 1, 4)
work_life_balance = st.slider("Work Life Balance (1 = Bad, 4 = Excellent)", 1, 4)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Convert input
input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "OverTime_Yes": 1 if overtime == "Yes" else 0
}

# Prepare input dataframe
input_df = pd.DataFrame([input_dict])

# Align columns with training data
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]

# Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Attrition (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Attrition (Probability: {probability:.2f})")