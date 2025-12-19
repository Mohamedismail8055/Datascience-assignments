import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the trained model and scaler
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
scaler = pickle.load(open('standard_scaler.pkl', 'rb'))

st.title('Titanic Survival Prediction')
st.write('Enter the passenger details to predict survival:')

# Input widgets for user data
pclass = st.selectbox('Pclass', [1, 2, 3])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('SibSp (Number of Siblings/Spouses Aboard)', 0, 8, 0)
parch = st.slider('Parch (Number of Parents/Children Aboard)', 0, 6, 0)
fare = st.number_input('Fare', value=30.0)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

if st.button('Predict Survival'):
    # Preprocess user input
    user_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [1 if sex == 'male' else 0],
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0]
    })

    # Ensure columns are in the same order as training data
    # The order of columns should match `features.columns` from the training notebook
    # For simplicity, we are assuming the order based on prior steps: Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S
    # A more robust way would be to save the feature columns from training and use them here.
    feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    user_data = user_data[feature_columns]

    # Scale the input features
    scaled_data = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display the result
    if prediction[0] == 1:
        st.success('Prediction: Survived!')
    else:
        st.error('Prediction: Not Survived.')
