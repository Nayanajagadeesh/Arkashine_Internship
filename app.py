from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


# Load the model and preprocessing objects
model = joblib.load('gpr_model.pkl')
imputer_X = joblib.load('imputer_X.pkl')
scaler_X = joblib.load('scaler_X.pkl')

def preprocess_user_input(user_input, imputer_X, scaler_X):
    user_input_imputed = imputer_X.transform(user_input)
    user_input_scaled = scaler_X.transform(user_input_imputed)
    return user_input_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = []

        for i in range(18):
            value = float(request.form[f'input_{i + 1}'])
            user_input.append(value)

        user_input_array = np.array([user_input])
        user_input_scaled = preprocess_user_input(user_input_array, imputer_X, scaler_X)

        predictions = model.predict(user_input_scaled)

        return render_template('results.html', predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)










