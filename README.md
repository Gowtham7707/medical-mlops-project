# Cardiovascular Disease MLflow Project (Windows)

This project includes:

* Dataset preprocessing
* Model training
* MLflow tracking
* Model saving
* Accuracy metrics
* Prediction API
* Beautiful Streamlit UI
* Real-time prediction flow

Dataset:

Cardiovascular Disease Dataset
[https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

---

# 1. Project Structure

```text
medical-mlflow-project/
│
├── data/
│   └── raw/
│       └── cardio_train.csv
│
├── models/
│   └── cardio_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   └── app.py
│
├── ui/
│   └── dashboard.py
│
├── requirements.txt
└── README.md
```

---

# 2. Create Virtual Environment

Open CMD:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

# 3. Install Packages

Create `requirements.txt`

```txt
mlflow
pandas
numpy
scikit-learn
matplotlib
fastapi
uvicorn
streamlit
joblib
requests
```

Install:

```bash
pip install -r requirements.txt
```

---

# 4. Download Dataset

Download:
[https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

Extract:

```text
cardio_train.csv
```

Place here:

```text
medical-mlflow-project/data/raw/cardio_train.csv
```

---

# 5. Preprocessing Code

## File: src/preprocess.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):

    df = pd.read_csv(path, sep=';')

    # Remove ID column
    df.drop('id', axis=1, inplace=True)

    X = df.drop('cardio', axis=1)
    y = df['cardio']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
```

---

# 6. Model Training with MLflow

## File: src/train.py

```python
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from preprocess import load_data


X_train, X_test, y_train, y_test, scaler = load_data(
    '../data/raw/cardio_train.csv'
)

mlflow.set_experiment('Cardiovascular_Disease_Project')

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print('\nAccuracy:', accuracy)

    print('\nClassification Report:\n')
    print(classification_report(y_test, predictions))

    # MLflow logging
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 10)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.sklearn.log_model(model, 'cardio_model')

    # Save model
    joblib.dump(model, '../models/cardio_model.pkl')

    # Save scaler
    joblib.dump(scaler, '../models/scaler.pkl')

    print('\nModel saved successfully')
```

---

# 7. Run Training

Open terminal inside `src` folder:

```bash
cd src
python train.py
```

---

# 8. Start MLflow UI

From project root:

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

You will see:

* Accuracy
* Parameters
* Metrics
* Model artifacts
* Experiment history

---

# 9. FastAPI Prediction API

## File: api/app.py

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('../models/cardio_model.pkl')
scaler = joblib.load('../models/scaler.pkl')


@app.get('/')
def home():
    return {'status': 'Cardiovascular API Running'}


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
```

---

# 10. Run API

Open terminal inside `api` folder:

```bash
cd api
uvicorn app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 11. Beautiful Streamlit Dashboard

## File: ui/dashboard.py

```python
import streamlit as st
import requests

st.set_page_config(
    page_title='Cardiovascular AI Dashboard',
    layout='wide'
)

st.title('❤️ Cardiovascular Disease Prediction')

st.subheader('Enter Patient Information')

age = st.slider('Age (days)', 10000, 30000, 18000)
gender = st.selectbox('Gender', [1, 2])
height = st.slider('Height', 100, 250, 170)
weight = st.slider('Weight', 30, 200, 70)
ap_hi = st.slider('Systolic BP', 80, 250, 120)
ap_lo = st.slider('Diastolic BP', 40, 180, 80)
cholesterol = st.selectbox('Cholesterol', [1, 2, 3])
gluc = st.selectbox('Glucose', [1, 2, 3])
smoke = st.selectbox('Smoke', [0, 1])
alco = st.selectbox('Alcohol', [0, 1])
active = st.selectbox('Physically Active', [0, 1])

if st.button('Predict Risk'):

    payload = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }

    response = requests.post(
        'http://127.0.0.1:8000/predict',
        json=payload
    )

    result = response.json()

    if result['prediction'] == 'High Risk':
        st.error(f"Prediction: {result['prediction']}")
    else:
        st.success(f"Prediction: {result['prediction']}")
```

---

# 12. Run Dashboard

Open terminal inside `ui` folder:

```bash
cd ui
streamlit run dashboard.py
```

---

# 13. Final Application Flow

```text
Dataset
   ↓
Preprocessing
   ↓
RandomForest Training
   ↓
MLflow Tracking
   ↓
Model Save
   ↓
FastAPI Prediction
   ↓
Streamlit Dashboard
```

---

# 14. Example Input

```json
{
  "age": 18000,
  "gender": 1,
  "height": 170,
  "weight": 70,
  "ap_hi": 160,
  "ap_lo": 100,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

---

# 15. Example Output

```json
{
  "prediction": "High Risk"
}
```

---

# 16. Next Improvements

You can later add:

* Docker
* Kubernetes
* Kubeflow
* KServe
* Real-time streaming
* ECG signals
* Grafana monitoring
* MQTT telemetry
* ONNX edge deployment

---

# 17. Useful Commands

## Start MLflow

```bash
mlflow ui
```

## Start API

```bash
uvicorn app:app --reload
```

## Start Dashboard

```bash
streamlit run dashboard.py
```

---

# 18. Recommended Order

1. Download dataset
2. Preprocess data
3. Train model
4. Start MLflow
5. Start API
6. Start dashboard
7. Test prediction
