from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)


CORS(app)

stunted_model = joblib.load('stunted.pkl')
nutrion_model = joblib.load('nutrion.pkl')


@app.route("/")
def hello_world():
    return "<p>Halo, Welcome!</p>"


@app.route("/predict", methods=['POST'])
def predict_anak():
    data = request.json

    jenis_kelamin = 0 if data["gender"] == 'Male' else 1


    result = ""

    stunting_klasifikasi = stunted_model.predict([[data["months"], jenis_kelamin, data["tb"]]])
    status_gizi = nutrion_model.predict([[data["months"], data["bb"], jenis_kelamin]])

    if stunting_klasifikasi[0] == 0 or stunting_klasifikasi[0] == 1:
        result = 'stunted'
    elif stunting_klasifikasi[0] == 2 or stunting_klasifikasi[0] == 3:
        result = 'normal'

    if status_gizi[0] == 0 or status_gizi[0] == 1:
        result = 'underweight'
    elif status_gizi[0] == 2:
        result = 'normal'
    elif status_gizi[0] == 3:
        result = 'overweight'

    return jsonify({
        "message": "data has been generated successfully",
        "data": result
    })
