from flask import Flask, render_template, request
import pickle
import numpy as np

# Load your trained machine learning model
# Assuming you've saved it using pickle
model = pickle.load(open('model.pkl', 'rb'))
ss = pickle.load(open('ss.pkl','rb'))

# Initialize Flask app
app = Flask(__name__)

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that processes form input
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = request.form['year']
        present_price = request.form['present price']
        kms_driven = request.form['kms driven']
        fuel_type = request.form['fuel type']
        seller_type = request.form['seller type']
        transmission_type = request.form['transmission']

        # Create an array of inputs for the model (reshape for single prediction)
        input_features = np.array([[year,present_price,kms_driven,fuel_type,seller_type,transmission_type,0]])
        input_features=ss.transform(input_features)

        # Make the prediction using the model
        prediction = model.predict(input_features)[0]

        # Render the prediction result
        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
