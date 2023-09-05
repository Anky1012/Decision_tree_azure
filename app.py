from flask import Flask ,render_template,request,app
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

Scaler=pickle.load(open("model/standardScalar.pkl","rb"))
model=pickle.load(open("model/modelForPrediction.pkl","rb"))

## route for Homepage


@app.route("/")
def index():
    return render_template('index.html')



## Route for single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure =float(request.form.get('BloodPressure'))
        SkinThickness =float(request.form.get('SkinThickness'))
        Insuline=float(request.form.get('Insuline'))
        BMI=float(request.form.get('BMI'))
        Diabetespedigreefunction=float(request.form.get('Diabetespedigreefunction'))
        Age=float(request.form.get('Age'))
        

        new_data =Scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insuline,BMI,Diabetespedigreefunction,Age]])

        predict=model.predict(new_data)

        if predict[0]==1 :
            result = 'Diabetic'

        else:
            result = "non-Diabetic"  

        return render_template('single_prediction.html',result=result)      
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")        