import numpy as np
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#Loading the saved model
loaded_model = pickle.load(open('D:/MLprojectDeploymentApplication/autismPrediction/model.sav','rb'))

st.title("Autism Prediction")
#User Inputs
st.header("Enter the details:")

A1_score = st.selectbox("A1_score: ",[0,1])
A2_score = st.selectbox("A2_score: ",[0,1])
A3_score = st.selectbox("A3_score: ",[0,1])
A4_score = st.selectbox("A4_score: ",[0,1])
A5_score = st.selectbox("A5_score: ",[0,1])
A6_score = st.selectbox("A6_score: ",[0,1])
A7_score = st.selectbox("A7_score: ",[0,1])
A8_score = st.selectbox("A8_score: ",[0,1])
A9_score = st.selectbox("A9_score: ",[0,1])
A10_score = st.selectbox("A10_score: ",[0,1])

#Inputs to be label encoded
gender = st.selectbox("Gender:", ["m", "f"])

ethnicity = st.selectbox("Ethnicity:", ['White-European', 'Middle Eastern ', 'Pasifika', 'Black',
       'Others', 'Hispanic', 'Asian', 'Turkish', 'South Asian', 'Latino'])

jaundice = st.radio("Jaundice at birth:", ["yes", "no"])

austim = st.radio("Family history of autism:", ["yes", "no"])

contry_of_res = st.selectbox("Country of resident: ",['Austria', 'India', 'United States', 'South Africa', 'Jordan',
       'United Kingdom', 'Brazil', 'New Zealand', 'Canada', 'Kazakhstan',
       'United Arab Emirates', 'Australia', 'Ukraine', 'Iraq', 'France',
       'Malaysia', 'Vietnam', 'Egypt', 'Netherlands', 'Afghanistan',
       'Oman', 'Italy', 'Unites States', 'Bahamas', 'Saudi Arabia',
       'Ireland', 'Aruba', 'Sri Lanka', 'Russia', 'Bolivia', 'Azerbaijan',
       'Armenia', 'Serbia', 'Ethiopia', 'Sweden', 'Iceland', 'Hong Kong',
       'Angola', 'China', 'Germany', 'Spain', 'Tonga', 'Pakistan', 'Iran',
       'Argentina', 'Japan', 'Mexico', 'Nicaragua', 'Sierra Leone',
       'Czech Republic', 'Niger', 'Romania', 'Cyprus', 'Belgium',
       'Burundi', 'Bangladesh'])

used_app_before = st.radio("used_ap_before:", ["yes", "no"])

relation = st.radio("relation:", ['Self', 'Others'])

object_columns = np.array([gender,ethnicity,jaundice,austim,contry_of_res,used_app_before,relation])

label_encoder = pickle.load(open('D:/MLprojectDeploymentApplication/autismPrediction/encoders.pkl','rb'))

# Encoding individual columns
gender_encoded = label_encoder['gender'].transform([gender])[0]
ethnicity_encoded = label_encoder['ethnicity'].transform([ethnicity])[0]
jaundice_encoded = label_encoder['jaundice'].transform([jaundice])[0]
austim_encoded = label_encoder['austim'].transform([austim])[0]
contry_of_res_encoded = label_encoder['contry_of_res'].transform([contry_of_res])[0]
used_app_before_encoded = label_encoder['used_app_before'].transform([used_app_before])[0]
relation_encoded = label_encoder['relation'].transform([relation])[0]

age = st.slider("Age:", 1, 100, 50)

result = st.number_input("Result:", min_value=-6.0, max_value=24.0, value=0.0, step=0.000001)

if st.button("Predict"):
    features = np.array([[A1_score,A2_score,A3_score,A4_score,
                        A5_score,A6_score,A7_score,A8_score,
                        A9_score,A10_score,age,gender_encoded,ethnicity_encoded,
                        jaundice_encoded,austim_encoded,contry_of_res_encoded,
                        used_app_before_encoded,result,relation_encoded]])
    
    prediction = loaded_model.predict(features)

    result = "Autistic" if prediction == 1 else "Not Autistic"
    st.subheader(f"Prediction: {result}")


#streamlit run web_page.py
