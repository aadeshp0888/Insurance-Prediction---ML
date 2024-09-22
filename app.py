from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    # Reshape and make prediction
    age = np.array([[age]])
    prediction = model.predict(age)[0]
    
    result = "Will buy insurance" if prediction == 1 else "Will not buy insurance"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
