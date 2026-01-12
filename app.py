import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder

# Load the pre-trained model
model = tf.keras.models.load_model('churn_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoded_geo = pickle.load(open('encode_geo.pkl', 'rb'))  # Contains OneHotEncoder for Geography
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))  # Contains LabelEncoder for Gender

st.title("Customer Churn Prediction Using ANN")

geography = st.selectbox("Select Geography", label_encoded_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", 18, 100, 30)
balance = st.number_input("Enter Balance")
credit_score = st.number_input("Enter Credit Score")
estimated_salary = st.number_input("Enter Estimated Salary")
tenure = st.slider("Select Tenure (Years with Bank)", 0, 10, 1)
num_of_products = st.slider("Select Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

input_data ={
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Create DataFrame from input data
input_data_df = pd.DataFrame(input_data)

# 1. Gender (using label_encoder_gender - LabelEncoder)  
input_data_df['Gender'] = label_encoder_gender.transform(input_data_df[['Gender']]).flatten()

# 2. Geography (using label_encoded_geo - OneHotEncoder)
geo_encoded = label_encoded_geo.transform(input_data_df[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoded_geo.get_feature_names_out(['Geography']))

# Drop original categorical columns and combine with encoded features
input_data_df = input_data_df.drop('Geography', axis=1)
input_data_df = pd.concat([input_data_df, geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data_df)

predict_button = st.button("Predict Churn")

if predict_button:
    prediction = model.predict(input_scaled)
    prediction_prob = prediction[0][0]
    st.write(f"Churn Probability: {prediction_prob:.2f}")
    if prediction_prob  > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")