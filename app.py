import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
import streamlit as st

# Load encoders and scaler
with open('label_encoder.pkl', 'rb') as obj:
    le = pickle.load(obj)

with open('onehot_encoder.pkl', 'rb') as obj:
    ohe = pickle.load(obj)

with open('scaler.pkl', 'rb') as obj:
    scaler = pickle.load(obj)

# Load model
model = load_model('model.h5')

st.title('Customer Churn Prediction')

# Input fields
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 0, 4)
has_cr_card = st.selectbox('Has Credit Card ?', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
    }

    df = pd.DataFrame(input_data, index=[0])

    # Transform categorical features
    df['Gender'] = le.transform(df['Gender'])

    encoded_geo = ohe.transform(df[['Geography']])
    encoded_df = pd.DataFrame(encoded_geo.toarray(), columns=ohe.get_feature_names_out(['Geography']))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop('Geography', axis=1, inplace=True)

    # Scale input
    input_scaled = scaler.transform(df)

    # Predict
    result = model.predict(input_scaled)

    # Show result
    if result > 0.5:
        st.error("⚠️ Customer is likely to **EXIT**.")
    else:
        st.success("✅ Customer will **NOT EXIT**.")
