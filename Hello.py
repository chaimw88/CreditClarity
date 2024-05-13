import streamlit as st
import pandas as pd
import pycaret
import pickle

# Load the trained model
loaded_model = pickle.load(open("test_cc", 'rb'))

def main():
    st.set_page_config(page_title="Prediction App", page_icon="🔍")

    st.title("Credit Score Prediction App")

    # Collect user inputs
    gender = st.selectbox('Gender', ['male', 'female'])
    number_of_children = st.number_input('Number of children', min_value=0, step=1)
    yearly_income = st.number_input('Yearly income', min_value=0.0)
    education_type = st.selectbox('Education type', [
        'Secondary / secondary special', 'Higher education', 'Lower secondary',
        'Incomplete higher', 'Academic degree'
    ])
    family_status = st.selectbox('Family status', [
        'Married', 'Civil marriage', 'Single / not married', 'Separated', 'Widow'
    ])
    income_type = st.selectbox('Income type', [
        'Commercial associate', 'Working', 'State servant', 'Pensioner', 'Student'
    ])
    housing_type = st.selectbox('Housing type', [
        'House / apartment', 'Rented apartment', 'With parents', 'Office apartment',
        'Municipal apartment', 'Co-op apartment'
    ])
    work_phone = st.selectbox('Work phone', ['yes', 'no'])
    phone = st.selectbox('Phone', ['yes', 'no'])
    email = st.selectbox('Email', ['yes', 'no'])
    occupation_type = st.selectbox('Occupation type', [
        'Drivers', 'HR staff', 'IT staff', 'Laborers', 'Managers', 'Pensioner',
        'Core staff', 'Accountants', 'Sales staff', 'Secretaries', 'Cooking staff',
        'Not Specified', 'Realty agents', 'Cleaning staff', 'Medicine staff',
        'Security staff', 'Low-skill Laborers', 'Waiters/barmen staff', 
        'High skill tech staff', 'Private service staff'
    ])
    number_of_family_members = st.number_input('Number of family members', min_value=1, step=1)
    own_car = st.selectbox('Own a car', ['yes', 'no'])
    own_realty = st.selectbox('Own realty', ['yes', 'no'])
    age = st.number_input('Age', min_value=0, step=1)
    days_employed_or_retired = st.number_input('Days employed / retired', min_value=0, step=1)
    employed = st.selectbox('Employed or not', ['yes', 'no'])

    # Convert categorical data to numerical
    categorical_mappings = {
        'yes': 1,
        'no': 0,
        'male': 0,
        'female': 1,
        'Secondary / secondary special': 0,
        'Higher education': 1,
        'Lower secondary': 2,
        'Incomplete higher': 3,
        'Academic degree': 4,
        'Married': 0,
        'Civil marriage': 1,
        'Single / not married': 2,
        'Separated': 3,
        'Widow': 4,
        'Commercial associate': 0,
        'Working': 1,
        'State servant': 2,
        'Pensioner': 3,
        'Student': 4,
        'House / apartment': 0,
        'Rented apartment': 1,
        'With parents': 2,
        'Office apartment': 3,
        'Municipal apartment': 4,
        'Co-op apartment': 5,
        'Drivers': 0,
        'HR staff': 1,
        'IT staff': 2,
        'Laborers': 3,
        'Managers': 4,
        'Pensioner': 5,
        'Core staff': 6,
        'Accountants': 7,
        'Sales staff': 8,
        'Secretaries': 9,
        'Cooking staff': 10,
        'Not Specified': 11,
        'Realty agents': 12,
        'Cleaning staff': 13,
        'Medicine staff': 14,
        'Security staff': 15,
        'Low-skill Laborers': 16,
        'Waiters/barmen staff': 17,
        'High skill tech staff': 18,
        'Private service staff': 19
    }

    data = {
        'CODE_GENDER': categorical_mappings[gender],
        'CNT_CHILDREN': number_of_children,
        'AMT_INCOME_TOTAL': yearly_income,
        'NAME_EDUCATION_TYPE': categorical_mappings[education_type],
        'NAME_FAMILY_STATUS': categorical_mappings[family_status],
        'NAME_INCOME_TYPE': categorical_mappings[income_type],
        'NAME_HOUSING_TYPE': categorical_mappings[housing_type],
        'FLAG_WORK_PHONE': categorical_mappings[work_phone],
        'FLAG_PHONE': categorical_mappings[phone],
        'FLAG_EMAIL': categorical_mappings[email],
        'OCCUPATION_TYPE': categorical_mappings[occupation_type],
        'CNT_FAM_MEMBERS': number_of_family_members,
        'OWN_CAR': categorical_mappings[own_car],
        'OWN_REALTY': categorical_mappings[own_realty],
        'AGE': age,
        'DAYS_EMPLOYED': days_employed_or_retired,
        'EMPLOYED_OR_NOT': categorical_mappings[employed]
    }

    input_df = pd.DataFrame([data])

    st.write(input_df)

    if st.button("Predict"):
        prediction = loaded_model.predict(input_df)
        st.write(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
