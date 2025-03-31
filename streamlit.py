import streamlit as st
import requests

st.title("Insurance Cross Sell Prediction")

Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
HasDrivingLicense = st.selectbox("Has Driving License", [0, 1])
RegionID = st.number_input("Region ID", min_value=1, max_value=100, value=25)
Switch = st.number_input("Switch", min_value=0, max_value=10, value=2)
PastAccident = st.text_input("Past Accident (NaN if not applicable)", "NaN")
AnnualPremium = st.number_input("Annual Premium", max_value=500000, value=210124)

Gender = "Male" if Gender == "Male" else "Female"
PastAccident = "NaN" if PastAccident.lower() == "nan" else str(PastAccident)

input_data = {
    "Gender": Gender,
    "Age": Age,
    "HasDrivingLicense": HasDrivingLicense,
    "RegionID": RegionID,
    "Switch": Switch,
    "PastAccident": PastAccident,
    "AnnualPremium": AnnualPremium
}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8080/predict", json=input_data)
    if response.status_code == 200:
        prediction = response.json()["predicted_class"]
        print(prediction)
        st.write(f"Prediction: {'Will Buy Insurance' if prediction == 1 else 'Will Not Buy Insurance'}")
    else:
        st.write("Error in prediction!")

