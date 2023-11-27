#!/usr/bin/env python
# coding: utf-8


from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load pickle files for the model as well as the labelencoder objects for City, State and maker
model = pickle.load(open('model.p','rb'))
city_label = pickle.load(open('city_label.p','rb'))
state_label = pickle.load(open('state_label.p','rb'))
make_label = pickle.load(open('make_label.p','rb'))

@app.route('/')
def index():
    return render_template('input.html')                                   # render the input html template upon opening the url

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [x for x in request.form.values()]                        # Get predictor variable values from the html input form
    city = city_label[input_data[2]]                                       # Parse the categorical City data into numerical to pass to model
    state = state_label[f' {input_data[3]}']                               # Parse the categorical State data into numerical to pass to model
    maker = make_label[input_data[4]]                                      # Parse the categorical maker data into numerical to pass to model
    input_data[2] = city
    input_data[3] = state
    input_data[4] = maker
    input_data = [int(x) for x in input_data]
    input_data = [np.array(input_data)]
    prediction = model.predict(input_data)                                 # Predict the price based on passed vallues
    prediction = round(prediction[0],2)
    return render_template('input.html', pred_text = f'The Car should be priced: {prediction} $')       # Output the price back to html form

if __name__ == '__main__':
    app.run(debug=True)

