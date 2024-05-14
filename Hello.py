import streamlit as st
import pandas as pd
import pycaret
import pickle

# Load the trained model
loaded_model = pickle.load(open("old_cc", 'rb'))

def main():
    st.set_page_config(page_title="Bank Credit Risk and Product Offering App", page_icon="🏦", layout="wide")

    # CSS to style the app
    st.markdown("""
        <style>
            body {
                background-color: black;
                color: white;
            }
            .stApp {
                background-color: black;
            }
            h1, h2, h3, h4, h5, h6, p, div, label {
                color: white;
            }
            .css-1r6slb0 {
                color: black;
            }
            .low-risk {
                color: green;
            }
            .high-risk {
                color: red;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>Bank Credit Risk Assessment and Product Offering</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
            Use this application to assess your credit risk and discover eligible bank products based on your profile.
            Please fill in the information below:
        </div>
    """, unsafe_allow_html=True)

    with st.form("user_input_form"):
        cols = st.columns((1, 1, 1))
        
        with cols[0]:
            st.subheader("Personal Information")
            gender = st.radio('Gender', ['Male', 'Female'])
            age = st.slider('Age', min_value=18, max_value=100, value=30, step=1)
            months_with_bank = st.slider('Months with Bank', min_value=0, max_value=300, value=24, step=1)
        
        with cols[1]:
            st.subheader("Financial Information")
            own_car = st.radio('Do you own a car?', ['Yes', 'No'])
            own_realty = st.radio('Do you own real estate?', ['Yes', 'No'])
            number_of_children = st.slider('Number of Children', min_value=0, max_value=10, value=1, step=1)
            yearly_income = st.number_input('Yearly Income (in USD)', min_value=0.0, step=1000.0, format='%f')
        
        with cols[2]:
            st.subheader("Background Information")
            days_employed_or_retired = st.number_input('Days Employed or Retired', min_value=0, step=1, format='%d')
            employed = st.radio('Are you currently employed?', ['Yes', 'No'])
            education_type = st.selectbox('Education Level', [
                'Secondary / Secondary Special', 'Higher Education', 'Lower Secondary',
                'Incomplete Higher', 'Academic Degree'
            ])
            family_status = st.selectbox('Family Status', [
                'Married', 'Civil Marriage', 'Single / Not Married', 'Separated', 'Widow'
            ])
            income_type = st.selectbox('Income Type', [
                'Commercial Associate', 'Working', 'State Servant', 'Pensioner', 'Student'
            ])
            housing_type = st.selectbox('Housing Type', [
                'House / Apartment', 'Rented Apartment', 'Living with Parents', 'Office Apartment',
                'Municipal Apartment', 'Co-op Apartment'
            ])
            contact_info = st.radio('Do you have both a phone and email?', ['Yes', 'No'])

        submit_button = st.form_submit_button(label='Assess Risk')

    if submit_button:
        if months_with_bank < 12:
            st.warning("Unable to assess risk as banking history is less than 12 months.")
        else:
            data = {
                'CODE_GENDER': gender,
                'CNT_CHILDREN': number_of_children,
                'AMT_INCOME_TOTAL': yearly_income,
                'NAME_EDUCATION_TYPE': education_type,
                'NAME_FAMILY_STATUS': family_status,
                'NAME_INCOME_TYPE': income_type,
                'NAME_HOUSING_TYPE': housing_type,
                'FLAG_PHONE': 1 if contact_info == 'Yes' else 0,
                'FLAG_EMAIL': 1 if contact_info == 'Yes' else 0,
                'OWN_CAR': 1 if own_car == 'Yes' else 0,
                'OWN_REALTY': 1 if own_realty == 'Yes' else 0,
                'AGE': age,
                'DAYS_EMPLOYED': days_employed_or_retired,
                'EMPLOYED_OR_NOT': 1 if employed == 'Yes' else 0
            }

            input_df = pd.DataFrame([data])
            input_df = ensure_all_features(input_df, loaded_model)
            assess_risk_and_offer_products(input_df, own_car, own_realty, yearly_income)

def ensure_all_features(input_df, model):
    dummies_df = pd.get_dummies(input_df)
    missing_cols = set(model.feature_names_in_) - set(dummies_df.columns)
    for col in missing_cols:
        dummies_df[col] = 0
    return dummies_df[model.feature_names_in_]

def assess_risk_and_offer_products(input_df, own_car, own_realty, yearly_income):
    prediction = loaded_model.predict(input_df)
    prediction_prob = loaded_model.predict_proba(input_df)[0][1]
    if prediction_prob > 0.40:
        risk_status = "low credit risk"
        risk_class = "low-risk"
    else:
        risk_status = "high credit risk"
        risk_class = "high-risk"
    
    st.markdown(f"### Prediction Result: <span class='{risk_class}'>{risk_status}</span>", unsafe_allow_html=True)

    if prediction_prob > 0.40:
        offer_products(own_car, own_realty, yearly_income)
    else:
        suggest_improvements()

def offer_products(own_car, own_realty, yearly_income):
    st.subheader("Eligible Product Offerings")
    product_offerings = []

    if own_car == 'No':
        product_offerings.append("- **Car Loan:** Consider our competitive car loan rates!")
    if own_realty == 'No':
        product_offerings.append("- **Mortgage:** Unlock home ownership with our tailored mortgages.")
    if yearly_income > 2000:
        product_offerings.append("- **Personal Loan:** You may qualify for a higher personal loan amount.")

    if product_offerings:
        for offering in product_offerings:
            st.markdown(offering)
    else:
        st.markdown("Currently, no product offerings available.")

def suggest_improvements():
    st.error("You are currently at a high credit risk. Here are some suggestions to improve your profile:")
    st.markdown("- We can offer a **credit coach** to help you get back on track.")

if __name__ == "__main__":
    main()
