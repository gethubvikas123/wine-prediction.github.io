import streamlit as st
import numpy as np

import pandas as pd
import joblib

model = joblib.load('Wine_prediction.joblib', mmap_mode=None)



def predictive_data(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction=model.predict(input_data_reshaped)
    print(prediction)
    if (prediction [0]==1):
        return("Good Quality Wine")
    else:
        return("Bad Quality Wine")

def main():
    st.title('Wine Prediction App')

    fixed_acidity = st.number_input("Enter Fixed acidity value")
    volatile_acidity = st.number_input("Enter volatile_acidity value")
    citric_acid = st.number_input("Enter citric_acid value")
    residual_sugar=st.number_input("Enter residual_sugar value")
    chlorides=st.number_input("Enter chlorides value")
    free_sulfur_dioxide=st.number_input("Enter ree_sulfur_dioxide value")
    total_sulfur_dioxide=st.number_input("Enter total_sulfur_dioxide value")
    density =st.number_input("Enter density  value")
    pH=st.number_input("Enter pH value")
    sulphates =st.number_input("Enter sulphates value")
    alcohol=st.number_input("Enter alcohol value")
    id=st.number_input("Enter Wine Id")

    prediction_value= ''

    if st.button("Check Quality"):

        prediction_value= predictive_data([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,pH, sulphates, alcohol,id])
        
        st.success(prediction_value)

if __name__=='__main__':
    main()

     
