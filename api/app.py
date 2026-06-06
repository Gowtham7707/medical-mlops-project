from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import numpy as np

app = FastAPI()
Instrumentator().instrument(app).expose(app)

model = joblib.load('models/cardio_model.pkl')
scaler = joblib.load('models/scaler.pkl')


@app.get('/')
def home():
    return {'status': 'Cardiovascular API Backend Running'}


@app.post('/predict')
def predict(data: dict):

    values = np.array([
        data['age'],
        data['gender'],
        data['height'],
        data['weight'],
        data['ap_hi'],
        data['ap_lo'],
        data['cholesterol'],
        data['gluc'],
        data['smoke'],
        data['alco'],
        data['active']
    ]).reshape(1, -1)

    values = scaler.transform(values)

    prediction = model.predict(values)[0]

    risk = 'High Risk' if prediction == 1 else 'Low Risk'

    return {
        'prediction': risk
    }