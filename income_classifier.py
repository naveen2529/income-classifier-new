import streamlit as st
import pandas as pd
import joblib

# Load pipeline (preprocessing + model)
income_pipeline = joblib.load("final_catboost_pipeline.pkl")

# Title
st.title("Income Prediction App")
st.markdown("Enter your details to predict if income is **>50K** or **<=50K**")

# Categorical options (based on training data)
income_workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                            'State-gov', 'Without-pay', 'Never-worked']
income_education_options = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
                            'Assoc-voc', 'Doctorate', 'Prof-school', '7th-8th', '12th', '1st-4th']
income_marital_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent']
income_occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                             'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                             'Farming-fishing', 'Transport-moving', 'Priv-house-serv']
income_relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
income_race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
income_sex_options = ['Male', 'Female']
income_native_country_options = ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada']

# Input fields
age = st.slider("Age", 18, 90, 30)
fnlwgt = st.number_input("Fnlwgt", value=100000, step=1000)
education_num = st.slider("Education Number", 1, 16, 10)
capital_gain = st.slider("Capital Gain", 0, 100000, 0)
capital_loss = st.slider("Capital Loss", 0, 5000, 0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

workclass = st.selectbox("Workclass", income_workclass_options)
education = st.selectbox("Education", income_education_options)
marital_status = st.selectbox("Marital Status", income_marital_options)
occupation = st.selectbox("Occupation", income_occupation_options)
relationship = st.selectbox("Relationship", income_relationship_options)
race = st.selectbox("Race", income_race_options)
sex = st.selectbox("Sex", income_sex_options)
native_country = st.selectbox("Native Country", income_native_country_options)

# Input DataFrame
income_input_df = pd.DataFrame([{
    'age': age,
    'fnlwgt': fnlwgt,
    'education-num': education_num,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'workclass': workclass,
    'education': education,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'native-country': native_country
}])

# Prediction
if st.button("Predict Income"):
    prediction = income_pipeline.predict(income_input_df)[0]
    proba = income_pipeline.predict_proba(income_input_df)[0]
    confidence = proba[prediction] * 100

    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"🧾 Predicted Income: **{label}**")
    st.info(f"🔢 Confidence: **{confidence:.2f}%**")
