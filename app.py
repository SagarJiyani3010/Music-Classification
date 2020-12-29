import pickle
import numpy as np, pandas as pd 
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
filename = 'models/model_gbc.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    data = np.array(data).reshape(1,-1)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    
    prediction = model.predict(data)
    return render_template('index.html', predict = prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
