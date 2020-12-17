from flask import Flask, request, jsonify 
import pickle
import pandas as pd
from flask_cors import CORS
from flask_cors import cross_origin

# untuk ngeload modelnya

filename = 'logistic_regression_tuned.sav'
#used_model = pickle.load(open(filename, 'rb')) 

try :
    used_model = pickle.load(open(filename, 'rb')) 
except Exception as e:
     print(e)

app = Flask(__name__)
CORS(app)

@app.route('/json_predict', methods=['POST'])
@cross_origin()
def json_predict():
    # ambil json
    json_predict = request.get_json()['bodyRequest']

    max_Age = 85 
    max_Region_Code = 52.0
    max_Annual_Premium = 61892.5
    if (int(json_predict['channel_binned']) >= 135):
        channel_binned = 1
    else:
        channel_binned = 0
    
    X_predict = {'Gender':[int(json_predict["gender"])],
                'Age':[int(json_predict['age']/max_Age)],
                'Driving_License':[int(json_predict['driving_license'])],
                'Region_Code':[float(json_predict['region_code']/max_Region_Code)],
                'Previously_Insured':[int(json_predict['previously_insured'])],
                'Vehicle_Damage':[int(json_predict['vehicle_damage'])],
                'Annual_Premium':[float(json_predict['annual_premium']/max_Annual_Premium)],
                'channel_binned':[channel_binned],
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
    app.run()
