import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('Project4KNN.pkl','rb')) 

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    
    r1 = float(request.args.get('Age'))
    r2 = float(request.args.get('SibSp'))
    r3 = float(request.args.get('Parch'))
    r4 = float(request.args.get('Fare'))
    r5 = float(request.args.get('Gender'))
    r6 = float(request.args.get('Pclass'))
    
    result=np.array([r1,r2,r3,r4,r5,r6]).reshape(1,-1)
    prediction = model.predict(result)
    
    if prediction == 0:
        return render_template('index.html', prediction_text='Luckely! The Person  Survive')
    else:
        return render_template('index.html', prediction_text='Unfortunateil! The Person Not Survive')

if __name__ == "__main__":
    app.run(debug=True)