import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "attrition_model.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_columns.pkl")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

st.title("🏢 توقع مغادرة الموظف (Employee Attrition)")
st.write("أدخل بيانات الموظف عشان تعرف احتمالية تركه للشركة")

st.header("بيانات الموظف")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("العمر", 18, 60, 30)
    monthly_income = st.number_input("الدخل الشهري", 1000, 20000, 5000)
    total_working_years = st.number_input("إجمالي سنين الخبرة", 0, 40, 5)
    years_at_company = st.number_input("سنين في الشركة الحالية", 0, 40, 3)
    distance_from_home = st.number_input("المسافة من البيت", 1, 30, 5)
    job_level = st.selectbox("المستوى الوظيفي", [1, 2, 3, 4, 5])
    overtime = st.selectbox("عمل إضافي (OverTime)", ["Yes", "No"])
    gender = st.selectbox("الجنس", ["Male", "Female"])

with col2:
    job_satisfaction = st.slider("الرضا الوظيفي", 1, 4, 3)
    work_life_balance = st.slider("التوازن بين الشغل والحياة", 1, 4, 3)
    environment_satisfaction = st.slider("الرضا عن بيئة العمل", 1, 4, 3)
    business_travel = st.selectbox("السفر للشغل", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("القسم", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("الدور الوظيفي", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    marital_status = st.selectbox("الحالة الاجتماعية", ["Single", "Married", "Divorced"])
    education_field = st.selectbox("مجال التعليم", [
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
    ])

if st.button("توقع الآن 🔍"):
    # ابني صف فاضي بكل أعمدة الموديل = 0
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # القيم الرقمية المباشرة
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['TotalWorkingYears'] = total_working_years
    input_data['YearsAtCompany'] = years_at_company
    input_data['DistanceFromHome'] = distance_from_home
    input_data['JobLevel'] = job_level
    input_data['JobSatisfaction'] = job_satisfaction
    input_data['WorkLifeBalance'] = work_life_balance
    input_data['EnvironmentSatisfaction'] = environment_satisfaction

    # Label Encoded columns
    if 'OverTime' in input_data.columns:
        input_data['OverTime'] = 1 if overtime == "Yes" else 0
    if 'Gender' in input_data.columns:
        input_data['Gender'] = 1 if gender == "Male" else 0

    # One-Hot columns
    def set_dummy(col_prefix, value):
        col_name = f"{col_prefix}_{value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    set_dummy("BusinessTravel", business_travel)
    set_dummy("Department", department)
    set_dummy("JobRole", job_role)
    set_dummy("MaritalStatus", marital_status)
    set_dummy("EducationField", education_field)

    # تأكد الترتيب مطابق لترتيب أعمدة الموديل وقت التدريب
    input_data = input_data[model_columns]

    proba = model.predict_proba(input_data)[:, 1][0]
    threshold = 0.2

    if proba >= 0.6:
        risk_level, color = "🔴 High Risk", "red"
    elif proba >= threshold:
        risk_level, color = "🟠 Medium Risk", "orange"
    else:
        risk_level, color = "🟢 Low Risk", "green"

    st.markdown("---")
    st.subheader("النتيجة")
    st.metric("احتمالية ترك الشركة", f"{proba:.1%}")
    st.markdown(f"### مستوى الخطورة: :{color}[{risk_level}]")
    st.progress(min(float(proba), 1.0))

    st.markdown("---")
    st.subheader("أهم العوامل المؤثرة في القرار (Feature Importance)")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': model_columns, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='#4C72B0')
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    st.pyplot(fig)
