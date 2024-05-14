import streamlit as st
import pandas as pd
import pickle

# Load the trained model
loaded_model = pickle.load(open("test_cc", 'rb'))

def main():
    st.set_page_config(page_title="1Bank Credit Risk and Product Offering App", page_icon="🏦", layout="wide")
    st.markdown("<h1 style='text-align: center; color: navy;'>Bank Credit Risk Assessment and Product Offering</h1>", unsafe_allow_html=True)
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
            yearly_income = st.number_input('Yearly Income (in USD)', min_value=0.0, step=500.0, format='%f')
        
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
                'FLAG_PHONE': contact_info,
                'FLAG_EMAIL': contact_info,
                'OWN_CAR': own_car,
                'OWN_REALTY': own_realty,
                'AGE': age,
                'DAYS_EMPLOYED': days_employed_or_retired,
                'EMPLOYED_OR_NOT': employed
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
    risk_status = "low credit risk" if prediction_prob > 0.40 else "high credit risk"
    st.markdown(f"### Prediction Result: {risk_status}")
    st.write(f"**Prediction Probability:** {prediction_prob:.2f}")

    if prediction_prob > 0.40:
        offer_products(own_car, own_realty, yearly_income)
    else:
        suggest_improvements(input_df)

def offer_products(own_car, own_realty, yearly_income):
    st.subheader("Eligible Product Offerings")
    if own_car == 'No':
        st.markdown("- **Car Loan:** Consider our competitive car loan rates!")
    if own_realty == 'No':
        st.markdown("- **Mortgage:** Unlock home ownership with our tailored mortgages.")
    if yearly_income > 50000:
        st.markdown("- **Personal Loan:** You may qualify for a higher personal loan amount.")

def suggest_improvements(input_df):
    st.error("You are currently at a high credit risk. Here are some suggestions to improve your profile:")
    test_income_modification(input_df)

def test_income_modification(input_df):
    original_value = input_df.loc[0, 'AMT_INCOME_TOTAL']
    original_prediction = loaded_model.predict_proba(input_df)[0][1]
    increments = [0.1, 0.2, 0.3]
    for increment in increments:
        modified_value = original_value * (1 + increment)
        input_df.at[0, 'AMT_INCOME_TOTAL'] = modified_value
        modified_prediction = loaded_model.predict_proba(input_df)[0][1]
        if modified_prediction > original_prediction:
            suggestion_text = f"Increasing **Yearly Income** from ${original_value:,.0f} to ${modified_value:,.0f} could decrease your credit risk."
            st.markdown("- " + suggestion_text)
    input_df.at[0, 'AMT_INCOME_TOTAL'] = original_value

if __name__ == "__main__":
    main()
