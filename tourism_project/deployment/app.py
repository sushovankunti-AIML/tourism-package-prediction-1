import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Sushovankunti/tourism-package-prediction", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourism package based on its operational parameters.
Please enter the input data below to get a prediction.
""")

# User input
Gender = st.selectbox("Gender", ["Male", "Female"])
TypeofContact = st.selectbox("TypeofContact", ['Company Invited', 'Self Inquiry', 'Phone Number'])
Occupation = st.selectbox("Occupation", ['Salaried', 'Self Employed', 'Business Owner'])
ProductPitched = st.selectbox("ProductPitched", ['Basic', 'Standard', 'Deluxe'])
MaritalStatus = st.selectbox("MaritalStatus", ['Married', 'Single', 'Divorced'])
Designation = st.selectbox("Designation", ['Executive', 'Managerial', 'Professional', 'Other'])
ProdToken = st.selectbox("ProdTaken", ['Yes', 'No'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("DurationofPitch (min)", min_value=1, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("NumberofPersonsVisiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("NumberofFollow-ups", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("NumberofTrips", min_value=0, max_value=10, value=2)
Passport = st.selectbox("Passport", ['Yes', 'No'])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("OwnCar", ['Yes', 'No'])
NumberOfChildrenVisiting = st.number_input("NumberofChildrenVisiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=50000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Gender': Gender,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProdToken': ProdToken,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
