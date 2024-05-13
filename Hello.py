import streamlit as st
import pandas as pd
import pycaret
import pickle

# Load the trained model
loaded_model = pickle.load(open("test_cc", 'rb'))

def main():
    st.set_page_config(page_title="Prediction App", page_icon="🔍")

    st.title("Data Collection and Prediction App")

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
    years_employed_or_retired = st.number_input('Years employed / retired', min_value=0, step=1)
    employed = st.selectbox('Employed or not', ['yes', 'no'])

    # Convert years employed to days
    days_employed_or_retired = years_employed_or_retired * 365

    # Create a dictionary with the input data
    data = {
        'CODE_GENDER': [gender],
        'CNT_CHILDREN': [number_of_children],
        'AMT_INCOME_TOTAL': [yearly_income],
        'NAME_EDUCATION_TYPE': [education_type],
        'NAME_FAMILY_STATUS': [family_status],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_HOUSING_TYPE': [housing_type],
        'FLAG_WORK_PHONE': [work_phone],
        'FLAG_PHONE': [phone],
        'FLAG_EMAIL': [email],
        'OCCUPATION_TYPE': [occupation_type],
        'CNT_FAM_MEMBERS': [number_of_family_members],
        'OWN_CAR': [own_car],
        'OWN_REALTY': [own_realty],
        'AGE': [age],
        'DAYS_EMPLOYED': [days_employed_or_retired],
        'EMPLOYED_OR_NOT': [employed]
    }

    input_df = pd.DataFrame(data)

    # Convert categorical variables to dummy/indicator variables
    input_df = pd.get_dummies(input_df)

    # Align the input_df with the columns used during training
    model_columns = loaded_model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    st.write("Input Data")
    st.write(input_df)

    if st.button("Predict"):
        # Make prediction
        prediction = loaded_model.predict(input_df)
        prediction_prob = loaded_model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.success("The person is eligible for a credit card.")
        else:
            st.error("The person is not eligible for a credit card unfortunately.")

        st.write(f"Prediction Score: {prediction_prob:.2f}")

        # Suggestions for improvement
        suggestions = []
        if yearly_income < 50000:  # Example threshold
            suggestions.append("Increase yearly income.")
        if age < 21:
            suggestions.append("Applicant should be older than 21.")
        if number_of_children > 3:
            suggestions.append("Reduce number of dependents.")
        if not (email == 'yes' and phone == 'yes'):
            suggestions.append("Ensure to have both phone and email contact details.")

        if suggestions:
            st.info("Suggestions for improvement:")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
        else:
            st.write("No suggestions for improvement.")

if __name__ == "__main__":
    main()
