from flask import Flask, request, jsonify 
import pickle
import pandas as pd
from flask_cors import CORS

# untuk ngeload modelnya

filename = 'logistic_regression_tuned.sav'
used_model = pickle.load(open(filename, 'rb')) 

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # nilai max df untuk normalisasi
    max_Age = 85 
    max_Region_Code = 52.0
    max_Annual_Premium = 61892.5

    # ngambil data input
    gender = int(request.form['gender'])
    age = int(request.form['age'])/max_Age
    driving_license = int(request.form['driving_license'])
    region_code = float(request.form['region_code'])/max_Region_Code
    previously_insured = int(request.form['previously_insured'])
    vehicle_damage = int(request.form['vehicle_damage'])
    annual_premium = float(request.form['annual_premium'])/max_Annual_Premium
    channel_binned = int(request.form['channel_binned'])
    mid = int(request.form['mid'])
    new = int(request.form['new'])
    old = int(request.form['old'])

    # bikin dataframe
    X_predict = {'Gender':[gender],
                'Age':[age],
                'Driving_License':[driving_license],
                'Region_Code':[region_code],
                'Previously_Insured':[previously_insured],
                'Vehicle_Damage':[vehicle_damage],
                'Annual_Premium':[annual_premium],
                'channel_binned':[channel_binned],
                'mid':[mid],
                'new':[new],
                'old':[old]}
    X_predict = pd.DataFrame(X_predict)
    
    # predict
    Y_predict = used_model.predict(X_predict)

    # ngembaliin nilai 0/1
    return Y_predict[0]

@app.route('/json_predict', methods=['POST'])
def json_predict():
    # ambil json
    json_predict = request.get_json()['bodyRequest']
    print(json_predict)

    X_predict = {'Gender':[int(json_predict["gender"])],
                'Age':[int(json_predict['age'])],
                'Driving_License':[int(json_predict['driving_license'])],
                'Region_Code':[float(json_predict['region_code'])],
                'Previously_Insured':[int(json_predict['previously_insured'])],
                'Vehicle_Damage':[int(json_predict['vehicle_damage'])],
                'Annual_Premium':[float(json_predict['annual_premium'])],
                'channel_binned':[int(json_predict['channel_binned'])],
                'mid':[int(json_predict['mid'])],
                'new':[int(json_predict['new'])],
                'old':[int(json_predict['old'])]}
    X_predict = pd.DataFrame(X_predict)

    # ubah jadidataframe
    # json_predict = pd.DataFrame([json_predict])

    # predict
    Y_predict = used_model.predict(X_predict)

    print('berhasil')

    # ngembaliin nilai 0/1
    return str(Y_predict[0])

if __name__ == "__main__":
    app.run(debug=True)
