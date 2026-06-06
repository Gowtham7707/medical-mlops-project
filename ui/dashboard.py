import streamlit as st
import requests
import os

st.set_page_config(
    page_title='Cardiovascular AI Dashboard',
    layout='wide'
)

st.title('❤️ Cardiovascular Disease Prediction')

st.subheader('Enter Patient Information')

# Read API URL dynamically
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict"
)

age = st.slider('Age (days)', 10000, 30000, 18000)
gender = st.selectbox('Gender', [1, 2])
height = st.slider('Height', 100, 250, 170)
weight = st.slider('Weight', 30, 200, 70)
ap_hi = st.slider('Systolic BP', 80, 250, 120)
ap_lo = st.slider('Diastolic BP', 40, 180, 80)
cholesterol = st.selectbox('Cholesterol', [1, 2, 3])
gluc = st.selectbox('Glucose', [1, 2, 3])
smoke = st.selectbox('Smoke', [0, 1])
alco = st.selectbox('Alcohol', [0, 1])
active = st.selectbox('Physically Active', [0, 1])

if st.button('Predict Risk'):

    payload = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    result = response.json()

    if result['prediction'] == 'High Risk':
        st.error(f"Prediction: {result['prediction']}")
    else:
        st.success(f"Prediction: {result['prediction']}")