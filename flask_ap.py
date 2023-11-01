from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the model
        input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        prediction = model.predict(input_data)

        # Map numerical prediction to class names
        species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species.get(prediction[0], 'Unknown')

        # Render the result page with the prediction
        return render_template('result.html', prediction=predicted_species)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
