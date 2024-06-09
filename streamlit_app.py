import streamlit as st
import joblib
import numpy as np

# Load the model and preprocessing objects
model = joblib.load('gpr_model.pkl')
imputer_X = joblib.load('imputer_X.pkl')
scaler_X = joblib.load('scaler_X.pkl')

def preprocess_user_input(user_input, imputer_X, scaler_X):
    user_input_imputed = imputer_X.transform(user_input)
    user_input_scaled = scaler_X.transform(user_input_imputed)
    return user_input_scaled

def main():
    st.title('Soil Property Prediction')

    # Input fields for user
    user_input = []
    for i in range(18):
        value = st.number_input(f'Input_{i + 1}', step=0.01)
        user_input.append(value)

    if st.button('Predict'):
        user_input_array = np.array([user_input])
        user_input_scaled = preprocess_user_input(user_input_array, imputer_X, scaler_X)
        predictions = model.predict(user_input_scaled)
        st.write('Predicted Values:')
        for i, pred in enumerate(predictions):
            st.write(f'{i + 1}. {pred}')

if __name__ == '__main__':
    main()
