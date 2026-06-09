---

Building an End-to-End MLOps Pipeline with MLflow, DVC, Docker, Kubernetes, KServe, Prometheus, Grafana, and GitHub Actions
Introduction
Machine Learning projects often begin as notebooks and experiments but eventually need to evolve into production-ready systems. A production MLOps platform requires much more than model training. It needs experiment tracking, dataset versioning, containerization, deployment automation, model serving, monitoring, observability, and CI/CD.
In this tutorial, we will build a complete Cardiovascular Disease Prediction MLOps project on Windows using:
MLflow for experiment tracking
DVC for dataset versioning
Docker for containerization
DockerHub for image registry
Kubernetes with Minikube
KServe for model serving
Prometheus for monitoring
Grafana for visualization
GitHub Actions for CI/CD automation

The dataset used in this project is the Cardiovascular Disease Dataset from Kaggle.

---

Final Production Architecture
The final system follows the architecture below:
GitHub
 ↓
GitHub Actions
 ↓
Train Model
 ↓
MLflow Tracking
 ↓
DVC Versioning
 ↓
Docker Build
 ↓
DockerHub Push
 ↓
KServe Deployment
 ↓
Prometheus Monitoring
 ↓
Grafana Dashboard
Every code change triggers an automated pipeline that trains the model, tracks experiments, versions datasets, builds Docker images, deploys to Kubernetes through KServe, and exposes monitoring metrics for Prometheus and Grafana.

---

Project Structure
Create the following project structure:
medical-mlops-project/
│
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml
│
├── api/
│   └── app.py
│
├── data/
│   └── raw/
│       └── cardio_train.csv
│
├── models/
│   ├── cardio_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── ui/
│   └── dashboard.py
│
├── k8s/
│   └── kserve-inference.yaml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── dvc.yaml
This structure separates data, source code, API code, UI code, deployment manifests, and CI/CD configuration.

---

Creating the Python Virtual Environment
Create a virtual environment to isolate dependencies.
python -m venv venv
Activate it:
venv\Scripts\activate
Using a virtual environment ensures project dependencies remain isolated from the system Python installation.

---

Installing Required Packages
Create a file named requirements.txt.
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
dvc
prometheus-fastapi-instrumentator
Install all dependencies:
pip install -r requirements.txt
These packages cover data processing, model training, experiment tracking, APIs, dashboards, monitoring, and dataset versioning.

---

Downloading the Dataset
Download the Cardiovascular Disease Dataset from Kaggle.
Extract the downloaded archive and locate:
cardio_train.csv
Place the dataset inside:
medical-mlops-project/data/raw/cardio_train.csv
The dataset contains patient information such as age, blood pressure, cholesterol levels, glucose levels, smoking habits, alcohol consumption, and physical activity indicators.

---

Dataset Versioning with DVC
Machine Learning datasets are usually too large to store directly in Git repositories. DVC solves this problem by versioning datasets while keeping the actual data outside Git.
Install DVC:
pip install dvc
Initialize DVC:
dvc init
Track the dataset:
dvc add data/raw/cardio_train.csv
DVC creates a metadata file:
cardio_train.csv.dvc
Git tracks this metadata file instead of the dataset itself.
Commit the changes:
git add .
git commit -m "Added DVC tracking"
With this setup, dataset changes become reproducible and version-controlled without bloating the Git repository.

---

Data Preprocessing
Create the file:
src/preprocess.py
The preprocessing pipeline performs:
Dataset loading
Feature extraction
Target extraction
Feature scaling
Train/Test splitting

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_data(path):
    df = pd.read_csv(path, sep=';')
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
This preprocessing pipeline standardizes all numerical features and prepares data for model training.

---

Training the Machine Learning Model
Create the file:
src/train.py
This script performs:
Data loading
Random Forest training
Evaluation
MLflow experiment logging
Model persistence

import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocess import load_data
X_train, X_test, y_train, y_test, scaler = load_data(
'data/raw/cardio_train.csv'
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
joblib.dump(model, 'models/cardio_model.pkl')
# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print('\nModel saved successfully')
The Random Forest algorithm is selected because it performs well on tabular healthcare datasets and provides strong baseline performance with minimal tuning.

---

Running the Training Pipeline
Execute the training script:
cd src
python train.py
Expected outputs include:
Accuracy score
Classification report
MLflow experiment records
Saved model artifact
Saved scaler artifact

The following files will be generated:
models/cardio_model.pkl
models/scaler.pkl
These artifacts will later be used by FastAPI and KServe for predictions.

---

Experiment Tracking with MLflow
Start the MLflow user interface:
mlflow ui
Open:
http://127.0.0.1:5000
MLflow provides:
Experiment tracking

Parameter comparison

Metric comparison

Model artifact storage

This allows data scientists and ML engineers to compare multiple model versions and identify the best-performing experiments before deployment.
In the next part, we will build the FastAPI prediction service, expose Prometheus metrics, create the Streamlit dashboard, containerize everything with Docker, and prepare the application for Kubernetes deployment.
Part 2: Building the Prediction API, Monitoring, Dashboard, and Docker Deployment
In Part 1, we built the machine learning training pipeline using:
DVC for dataset versioning
MLflow for experiment tracking
Scikit-learn for model training
Joblib for model persistence

At the end of the training stage, we generated:
models/cardio_model.pkl
models/scaler.pkl
Now we will expose the trained model through a production-ready API, add monitoring capabilities, create a user interface, and containerize the entire application.

---

Building the FastAPI Prediction Service
A trained model is useful only when applications can consume it.
For production deployments, FastAPI is an excellent choice because it provides:
High performance
Automatic OpenAPI documentation
Easy integration with machine learning models
Native support for asynchronous processing

Create:
api/app.py
Add the following implementation:
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import numpy as np
app = FastAPI()
Instrumentator().instrument(app).expose(app)
model = joblib.load('../models/cardio_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
@app.get('/')
def home():
    return {
        'status': 'Cardiovascular API Running'
    }
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
This service loads the trained model and scaler during startup, allowing predictions to be served instantly without retraining.

---

Understanding the Prediction Flow
When a request reaches the API:
Client Request
      ↓
FastAPI
      ↓
Input Features
      ↓
Scaler
      ↓
Random Forest Model
      ↓
Prediction
      ↓
Response
The API receives patient attributes, scales them using the saved scaler, generates predictions using the trained Random Forest model, and returns a risk assessment.

---

Enabling Prometheus Monitoring
Production systems require visibility.
Without monitoring, teams cannot answer questions such as:
How many requests are arriving?
What is the API latency?
Is CPU usage increasing?
Are requests failing?

The following line automatically exposes Prometheus metrics:
Instrumentator().instrument(app).expose(app)
This creates a metrics endpoint:
/metrics
Prometheus will later scrape this endpoint.

---

Running the FastAPI Application
Start the API:
uvicorn api.app:app --reload
After startup, open:
http://127.0.0.1:8000/docs

FastAPI automatically generates Swagger documentation.
You can now test prediction requests directly from the browser.

---

Accessing the Metrics Endpoint
Open:
http://127.0.0.1:8000/metrics
You should see Prometheus-compatible metrics.
Examples include:
http_requests_total
process_cpu_seconds_total
http_request_duration_seconds
These metrics become the foundation for production monitoring.

---

Building the Streamlit Dashboard
While APIs are useful for applications, users often need a graphical interface.
Streamlit allows us to create a dashboard with minimal effort.
Create:
ui/dashboard.py
Add the following code:
import streamlit as st
import requests
st.set_page_config(
    page_title='Cardiovascular AI Dashboard',
    layout='wide'
)
st.title('❤️ Cardiovascular Disease Prediction')
age = st.slider('Age', 10000, 30000, 18000)
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
        st.error(result['prediction'])
    else:
        st.success(result['prediction'])
The dashboard acts as a frontend client that communicates directly with the FastAPI backend.

---

Dashboard Communication Flow
The request path is:
User
 ↓
Streamlit Dashboard
 ↓
FastAPI API
 ↓
Random Forest Model
 ↓
Prediction Result
 ↓
Dashboard Display
This architecture separates the presentation layer from the prediction service.

---

Running the Dashboard
Start Streamlit:
streamlit run ui/dashboard.py
Open:
http://localhost:8501
You should see a cardiovascular disease prediction dashboard with sliders and dropdown fields.
Users can enter patient information and immediately receive a risk assessment.

---

Why Separate Frontend and Backend?
Many beginners combine everything into one application.
Production systems typically separate:
Frontend
Backend
Database
Monitoring
Model Serving
Benefits include:
Independent scaling
Easier maintenance
Better security
Easier deployments
Team ownership separation

This architecture mirrors enterprise-grade systems.

---

Containerizing the Application with Docker
The next step is packaging the application into a portable container.
Docker solves the classic problem:
"It works on my machine."
Containers package:
Code
Dependencies
Runtime
Configuration

into a reproducible artifact.

---

Creating the Dockerfile for Frontend.
Create:
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl

COPY ui ./ui

EXPOSE 8501

CMD ["streamlit", "run", "ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
Creating the Dockerfile for Backend.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

Docker compose
version: '3.9'

services:

  backend:

    build:
      context: .
      dockerfile: Dockerfile.api

    container_name: backend

    ports:
      - "8000:8000"

  frontend:

    build:
      context: .
      dockerfile: Dockerfile.ui

    container_name: frontend

    ports:
      - "8501:8501"

    environment:
      API_URL: "http://backend:8000/predict"

    depends_on:
      - backend
docker compose up --build

This image contains:
FastAPI application
Trained model
Scaler
Dependencies

Everything needed to serve predictions.

---

Understanding Dockerfile Instructions
Base Image
FROM python:3.11-slim
Provides a lightweight Python environment.
Working Directory
WORKDIR /app
Sets the default directory inside the container.
Dependency Installation
RUN pip install --no-cache-dir -r requirements.txt
Installs project packages.
Application Copy
COPY . .
Copies source code into the image.
Port Exposure
EXPOSE 8000
Makes the API port accessible.
Startup Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
Starts the FastAPI service automatically.

---

Building the Docker Image
Build the image:
docker build -t gowtham/cardio-mlops:v1 .
Docker creates an image containing the complete application stack.
Verify:
docker images
You should see:
gowtham/cardio-mlops:v1

---

Pushing Images to DockerHub
DockerHub acts as a centralized image registry.
Login:
docker login
Push the image:
docker push gowtham/cardio-mlops:v1

Once pushed, Kubernetes can pull and deploy the image from anywhere.

---

Why DockerHub Matters
Without a registry:
Developer Machine
      ↓
Manual Copy
      ↓
Server
With DockerHub:
Developer
      ↓
DockerHub
      ↓
Kubernetes
      ↓
Production
This creates a repeatable deployment pipeline.

---

Setting Up Minikube
To run Kubernetes locally on Windows, use Minikube.
Start the cluster:
minikube start --cpus 4 --memory 8192
If startup fails:
minikube delete
Remove-Item -Recurse -Force $HOME\.minikube
minikube start --cpus 4 --memory 8192
Minikube provides:
Local Kubernetes cluster
Kubernetes API Server
Scheduler
Controller Manager
Networking

all running on a local machine.

---

Why Kubernetes?
Docker solves packaging.
Kubernetes solves orchestration.
Docker provides:
Container
Kubernetes provides:
Deployment
Scaling
Self-Healing
Rolling Updates
Service Discovery
Load Balancing
This is why Kubernetes has become the standard platform for production machine learning deployments.

---

What's Next?
At this point we have:
✅ Trained ML model
✅ MLflow experiment tracking
✅ DVC dataset versioning
✅ FastAPI prediction service
✅ Prometheus metrics endpoint
✅ Streamlit frontend dashboard
✅ Docker containerization
✅ DockerHub image registry
✅ Minikube Kubernetes cluster
In Part 3, we will move into Kubernetes-native machine learning deployment by installing:
Helm
Istio
Knative
KServe

We will then deploy our model as a KServe InferenceService and understand how KServe, Knative, and Istio work together in a production MLOps platform.
Part 3: Deploying Machine Learning Models with Kubernetes, Istio, Knative, and KServe
In Part 2, we completed:
FastAPI prediction API
Prometheus metrics integration
Streamlit dashboard
Docker image creation
DockerHub image publishing
Minikube Kubernetes cluster setup

Now we move into the model serving layer of the MLOps platform.
This is where machine learning transitions from a containerized application into a scalable, cloud-native inference service powered by Kubernetes and KServe.

---

Why KServe?
Traditional deployments usually expose a model through a custom API.
Example:
Client
 ↓
FastAPI
 ↓
Model
While this works, enterprise environments require:
Autoscaling
Traffic routing
Canary deployments
Serverless inference
Service mesh integration
High availability

KServe provides these capabilities out of the box.

---

KServe Architecture
KServe relies on several Kubernetes technologies:
KServe
   │
   ├── Istio
   ├── Knative
   └── Kubernetes
Each component serves a specific purpose.
Kubernetes
Provides:
Scheduling
Scaling
Networking
Resource management

Istio
Provides:
Service mesh
Traffic routing
Security
Ingress management

Knative
Provides:
Serverless capabilities
Autoscaling
Scale-to-zero
Request queueing

KServe
Provides:
Machine learning model serving
Inference endpoints
Model lifecycle management

Together they create a production-grade model serving platform.

---

Installing Helm
Many Kubernetes components are distributed using Helm.
Helm acts as a package manager for Kubernetes.
Download Helm from the official release page. → Helm Releases
Download package:
helm-vX.X.X-windows-amd64.zip
Extract the archive.
Inside the package:
windows-amd64/helm.exe
Copy:
helm.exe
to:
C:\Windows\
or
C:\Program Files\Helm\
If using a custom folder, add it to the Windows PATH.
If using custom folder:
1. Open:
Environment Variables
2. Edit:
Path
3. Add:
C:\Program Files\Helm\
Verify installation:
helm version
Expected output should display the installed Helm version.

---

Why Helm?
Without Helm:
Install 100 Kubernetes YAML files manually
With Helm:
helm install application-name package
Helm simplifies deployment and upgrades of complex Kubernetes applications.

---

Installing IstioCTL
Istio requires a management tool called IstioCTL.
Download the Windows release package. → Istio Releases
Example:
istio-1.xx.x-win.zip
Extract to:
C:\istio
You should see:
bin\istioctl.exe
Add:
Add Istio to PATH
Add:
C:\istio\bin
to Windows PATH.
C:\istio\bin
to your PATH.
Verify:
istioctl version
Successful output confirms installation.

---

Installing KServe Dependencies
Before KServe can run, several Kubernetes components must be installed.

---

Install Cert Manager
Cert Manager manages certificates required by Kubernetes services.
Install:
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml

Verify:
kubectl get pods -n cert-manager

Wait until all pods are running.

---

Install Istio
Deploy Istio using:
istioctl install --set profile=demo -y

The demo profile is ideal for local development environments.
Verify:
kubectl get pods -n istio-system

All Istio components should be running.

---

Enable Automatic Sidecar Injection
Enable Istio injection:
kubectl label namespace default istio-injection=enabled

Why?
Because every deployed application automatically receives an Istio proxy sidecar.
This enables:
Traffic routing
Observability
Security
Service mesh communication

without modifying application code.

---

Installing Knative
Knative provides the serverless foundation required by KServe.
Install CRDs:
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.13.1/serving-crds.yaml

Install core components:
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.13.1/serving-core.yaml
Install networking layer:
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.15.1/net-istio.yaml
Verify:
kubectl get pods -n knative-serving
All pods should reach the Running state.

---

Understanding Knative
Knative introduces powerful serverless behavior.
Without Knative:
Pod Always Running
With Knative:
Request Arrives
        ↓
Pod Starts
        ↓
Serve Request
        ↓
Scale Up
        ↓
Scale Down
This reduces resource usage dramatically.

---

Installing KServe
Now install KServe itself.
Deploy:
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.13.1/kserve.yaml
Wait for deployment:
kubectl get pods -n kserve
All components should eventually become healthy.

---

Fixing ImagePullBackOff Issues
Sometimes the KServe controller fails to start because of image pull issues.
Update the controller image:
kubectl -n kserve set image deployment/kserve-controller-manager kube-rbac-proxy=quay.io/brancz/kube-rbac-proxy:v0.18.1
Restart deployment:
kubectl rollout restart deployment kserve-controller-manager -n kserve
Verify:
kubectl describe pod -n kserve
Check events and image pull status.

---

Creating the KServe Inference Service
Create:
k8s/kserve-inference.yaml
Add:
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService

metadata:
  name: cardio-model
spec:
  predictor:
    containers:
    - name: cardio-container
      image: gowtham/cardio-mlops:IMAGE_TAG
      ports:
      - containerPort: 8000
This tells KServe:
Create an inference endpoint
Deploy the specified Docker image
Expose prediction functionality
Manage scaling automatically

---

Deploying the Inference Service
Apply the resource:
kubectl apply -f k8s/kserve-inference.yaml
Check status:
kubectl get inferenceservice
or
kubectl get isvc
Example output:
cardio-model
Once Ready becomes True, the model is available.

---

Understanding the Generated Pods
KServe automatically creates deployment resources.
You may see something similar:
cardio-model-predictor-00001-deployment
A common observation is:
3/3 Running
Many engineers wonder why three containers exist inside one pod.

---

What Does 3/3 Running Mean?
KServe integrates:
Kubernetes
Knative
Istio

Therefore a single inference pod typically contains:
cardio-container
queue-proxy
istio-proxy
Each serves a unique purpose.

---

Container 1: Your Application
cardio-container
Contains:
FastAPI
Trained model
Prediction logic

This is the code we built earlier.

---

Container 2: Knative Queue Proxy
queue-proxy
Added automatically by Knative.
Responsibilities:
Request queueing
Autoscaling metrics
Scale-to-zero support
Traffic management

Knative monitors requests through this component.

---

Container 3: Istio Proxy
istio-proxy
Added through Istio sidecar injection.
Responsibilities:
Networking
Service mesh
Traffic routing
Secure communication
Ingress handling

This component enables advanced routing and observability.

---

Verifying Container Names
Inspect the pod:
kubectl describe pod cardio-model-predictor-00001-deployment
or:
kubectl get pod <pod-name> -o jsonpath="{.spec.containers[*].name}"
Example:
cardio-container queue-proxy istio-proxy
This confirms the KServe deployment architecture.

---

Internal Request Flow
When a client sends a prediction request:
Client
 ↓
Istio Gateway
 ↓
istio-proxy
 ↓
queue-proxy
 ↓
cardio-container
 ↓
Prediction Response
Notice that requests do not directly enter the application container.
Istio and Knative intercept requests first and provide enterprise-grade networking and scaling.

---

Building a Frontend for KServe
Deploy a frontend application that communicates directly with the KServe endpoint.
Example deployment:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend-kserve
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: gowthamc121/medical-mlflow-project-frontend
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: "http://cardio-model.default.svc.cluster.local/v1/models/cardio-model:predict"
The frontend communicates directly with the KServe inference endpoint.

---

Creating the Frontend Service
Expose the frontend:
apiVersion: v1
kind: Service
metadata:
  name: frontend-service-kserve
spec:
  selector:
    app: frontend
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
  type: NodePort
Apply:
kubectl apply -f kserve/frontend-kserve-service.yaml

Accessing the Frontend
Expose using Minikube:
minikube service frontend-service-kserve
Or port-forward:
kubectl port-forward service/frontend-service-kserve 8501:8501
Open:
http://localhost:8501
Your Streamlit application should now communicate with KServe rather than directly with FastAPI.

---

Complete KServe Communication Flow
Browser
 ↓
Frontend Dashboard
 ↓
KServe Endpoint
 ↓
Istio
 ↓
Knative
 ↓
FastAPI Container
 ↓
ML Model
 ↓
Prediction Result
This architecture mirrors how many modern cloud-native machine learning platforms deploy and serve models in production.

---

What We Have Accomplished
At this stage we now have:
✅ Kubernetes cluster
✅ Helm installed
✅ Istio service mesh
✅ Knative serverless platform
✅ KServe model serving
✅ InferenceService deployment
✅ Autoscaling architecture
✅ Frontend-to-KServe communication
✅ Production-grade inference endpoint
In Part 4, we will implement:
Prometheus installation
Metrics scraping
Grafana installation
Grafana dashboards
Visualization of FastAPI metrics
Kubernetes monitoring architecture
Observability best practices for MLOps systems

Part 4: Monitoring Machine Learning Systems with Prometheus and Grafana
In previous sections, we built:
MLflow experiment tracking
DVC dataset versioning
FastAPI prediction APIs
Streamlit dashboard
Docker containers
Kubernetes deployments
KServe inference services

The next critical component of a production MLOps platform is observability.
Without monitoring, teams cannot answer questions such as:
Is the prediction service healthy?
How many requests are arriving?
Is latency increasing?
Is CPU usage becoming excessive?
Are deployments working correctly?
Is autoscaling functioning?

This is where Prometheus and Grafana become essential.

---

Why Monitoring Matters
A machine learning model running in production is a live service.
Unlike training environments, production systems require continuous visibility.
Monitoring enables teams to:
Detect failures quickly
Identify performance bottlenecks
Measure request volumes
Track infrastructure health
Observe API response times
Validate scaling behavior

A typical production monitoring flow looks like:
FastAPI Application
          ↓
Prometheus Metrics
          ↓
Prometheus Server
          ↓
Grafana Dashboard
          ↓
Operations Team

---

Prometheus Overview
Prometheus is an open-source monitoring system.
Instead of applications pushing metrics, Prometheus pulls metrics by periodically scraping endpoints.
The workflow is:
Application
      ↓
/metrics
      ↓
Prometheus Scraping
      ↓
Time-Series Database
      ↓
Queries & Alerts
Prometheus stores all collected metrics as time-series data.
Examples:
CPU Usage
Memory Usage
Request Count
Request Duration
Error Rates

---

FastAPI Metrics Integration
Earlier, we added:
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
This automatically exposes:
/metrics
on the FastAPI application.
Verify:
http://localhost:8000/metrics
You should see metrics similar to:
http_requests_total
process_cpu_seconds_total
python_gc_objects_collected_total
http_request_duration_seconds
Prometheus will scrape this endpoint continuously.

---

Installing Prometheus with helm

Create the name space:
lm repo add prometheus-community https://prometheus-community.github.io/helm-charts
Add repo:
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
Install prometheus:

helm repo update
helm install prometheus prometheus-community/prometheus -n monitoring
In windows want to access means
kubectl port-forward svc/prometheus-server 9090:80 -n monitoring
Open Browser:
http://localhost:9090

Installing Prometheus ( optional)
Download Prometheus from the official release package.
Extract the archive.
Example structure:
prometheus/
│
├── prometheus.exe
├── promtool.exe
├── prometheus.yml
└── consoles/
Prometheus configuration is controlled through:
prometheus.yml

---

Configuring Prometheus
Open:
prometheus.yml
Replace contents with:
global:
  scrape_interval: 5s
scrape_configs:
- job_name: "fastapi"
  static_configs:
  - targets:
    - host.minikube.internal:8000
Explanation:
scrape_interval
scrape_interval: 5s
Prometheus collects metrics every 5 seconds.
job_name
job_name: "fastapi"
Logical name of the target.
targets
host.minikube.internal:8000
Location of the FastAPI service exposing metrics.

---

Starting Prometheus
Launch Prometheus:
prometheus.exe
Open:
http://localhost:9090
The Prometheus dashboard should appear.

---

Verifying Targets
Inside Prometheus:
Status
 ↓
Targets
You should see:
fastapi
UP
If status is:
DOWN
Verify:
FastAPI is running
Port 8000 is reachable
Metrics endpoint is accessible

---

Exploring Metrics
Navigate to:
Graph
Try:
http_requests_total
Press:
Execute
Prometheus will display collected data points.
Try:
process_cpu_seconds_total
and
http_request_duration_seconds_count
These metrics provide insight into API activity and performance.

---

Prometheus Architecture
The complete flow is:
FastAPI
    ↓
/metrics
    ↓
Prometheus
    ↓
Time-Series Storage
    ↓
PromQL Queries
Prometheus becomes the centralized monitoring engine.

---

Installing Grafana with helm
Add Helm Repo:
helm repo add grafana https://grafana.github.io/helm-charts
Update Helm:
helm repo update
Install Grafana:
helm install grafana grafana/grafana -n monitoring
Prometheus stores data.
Grafana visualizes it.
Get Grafana Admin Password:
$pwd = kubectl get secret -n monitoring grafana -o jsonpath="{.data.admin-password}"
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($pwd))
Username: admin
Password: <>
Port forwording:
kubectl port-forward svc/grafana 3000:80 -n monitoring
Default URL:
http://localhost:3000
You will be prompted to change the password after first login.

---

Grafana Overview
Grafana transforms raw metrics into dashboards.
Benefits include:
Interactive charts
Real-time visualization
Historical trends
Alerts
Team dashboards

Instead of reading raw Prometheus metrics:
http_requests_total 12983
Grafana displays intuitive graphs and panels.

---

Connecting Grafana to Prometheus
Inside Grafana:
Connections
 ↓
Data Sources
 ↓
Add Data Source
Select:
Prometheus

Configure:
URL:
http://prometheus-server.monitoring.svc.cluster.local

Click:
Save & Test

Expected result:
Data source is working
Grafana can now query Prometheus metrics.

---

Creating a Dashboard
Inside Grafana:
Dashboards
 ↓
Create Dashboard
 ↓
Add Visualization

Choose:
Prometheus
as the data source.

---

Request Count Visualization
Query:
http_requests_total
This panel displays:
Total requests
Request growth
Traffic spikes

Useful for measuring system usage.

---

CPU Usage Visualization
Query:
process_cpu_seconds_total
This panel displays:
CPU consumption
Resource trends
Capacity planning indicators

Useful for identifying performance bottlenecks.

---

Request Duration Visualization
Query:
http_request_duration_seconds_count
This panel displays:
Request volume
Latency behavior
Service responsiveness

Useful for detecting slow APIs.

---

Recommended Dashboard Layout
A common layout:
+---------------------+
| Total Requests      |
+---------------------+
+---------------------+
| CPU Usage           |
+---------------------+
+---------------------+
| Request Duration    |
+---------------------+

This provides a quick operational overview.

---

Monitoring Architecture
The complete monitoring architecture now becomes:
User Request
      ↓
FastAPI
      ↓
/metrics
      ↓
Prometheus
      ↓
Grafana
      ↓
Dashboard
Operations teams can monitor the application in real time.

---

Kubernetes Monitoring Flow
When deployed on Kubernetes:
FastAPI Pod
      ↓
Metrics Endpoint
      ↓
Prometheus
      ↓
Grafana
      ↓
Operations Dashboard
Monitoring becomes centralized regardless of the number of pods.

---

Why Prometheus and Grafana Are Industry Standards
Most enterprise MLOps platforms use this combination because:
Prometheus
Provides:
Metrics collection
Time-series storage
Alerting support
Kubernetes integration

Grafana
Provides:
Visualization
Dashboards
Alerts
Reporting

Together they form one of the most widely adopted observability stacks.

---

Production Monitoring Architecture
Our platform now looks like:
Browser
 ↓
Frontend Dashboard
 ↓
FastAPI API
 ↓
ML Model
 ↓
Prometheus Metrics
 ↓
Prometheus
 ↓
Grafana
 ↓
Monitoring Dashboard
This architecture allows both prediction serving and operational monitoring.

---

Key Takeaways
At this stage we have implemented:
✅ MLflow experiment tracking
✅ DVC dataset versioning
✅ FastAPI prediction APIs
✅ Streamlit frontend
✅ Docker containerization
✅ Kubernetes deployment
✅ KServe model serving
✅ Prometheus monitoring
✅ Grafana dashboards
✅ Production observability
In the next section, we will build the CI/CD layer using GitHub Actions, automate Docker image creation, publish images to DockerHub, deploy automatically to Kubernetes, and complete the end-to-end production MLOps pipeline.
Part 5: Automating Deployments with GitHub Actions and Building the Complete Production MLOps Pipeline
In the previous sections, we built:
MLflow experiment tracking
DVC dataset versioning
FastAPI prediction APIs
Streamlit dashboards
Docker containers
Kubernetes deployments
KServe model serving
Prometheus monitoring
Grafana dashboards

The final piece of the MLOps platform is CI/CD automation.
Without CI/CD, every deployment requires manual steps:
Developer
    ↓
Train Model
    ↓
Build Docker Image
    ↓
Push Docker Image
    ↓
Update Kubernetes
    ↓
Deploy
This process is error-prone and difficult to scale.
CI/CD allows deployments to happen automatically whenever code changes are pushed to GitHub.

---

What is CI/CD?
CI/CD stands for:
Continuous Integration (CI)
Automatically:
Validates code
Runs tests
Builds artifacts

Continuous Deployment (CD)
Automatically:
Builds Docker images
Pushes images to registries
Updates Kubernetes deployments
Deploys applications

The goal is:
Developer Push
      ↓
Automated Deployment

---

Why GitHub Actions?
GitHub Actions provides built-in automation directly inside GitHub repositories.
Benefits:
No external CI server required
Native GitHub integration
Docker suppor
Kubernetes deployment support
Secret management

Workflow execution occurs automatically whenever repository events occur.

---

CI/CD Architecture
Our final deployment workflow will look like:
GitHub Push
      ↓
GitHub Actions
      ↓
Build Docker Image
      ↓
Push DockerHub Image
      ↓
Update KServe Manifest
      ↓
Deploy Kubernetes Resources
      ↓
Production Environment
Every code change follows this path automatically.

---

Creating the GitHub Actions Workflow
Create:
.github/workflows/mlops-pipeline.yml
This file defines the complete deployment pipeline.

---

GitHub Actions Pipeline Configuration for windows machine
Add:
name: cardio-medical-mlflow-pipeline

on:
  push:
    branches:
        - master
  workflow_dispatch:      

jobs:
  deploy:
    runs-on: self-hosted
    environment: docker

    steps:
      - name: Checkout
        uses: actions/checkout@v4
 
      - name: Login DockerHub
        uses: docker/login-action@v3
        with: 
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build Backend Docker Image
        run: |
          docker build -f Dockerfile.api -t gowthamc121/medical-mlflow-project-backend:${{ github.sha }} .

      - name: Push Docker Image
        run: |
          docker push gowthamc121/medical-mlflow-project-backend:${{ github.sha }}

      - name: Replace Image Tag
        shell: powershell
        run: |
          (Get-Content kserve/kserve-inference.yaml) `
          -replace 'IMAGE_TAG','${{ github.sha }}' |
          Set-Content kserve/kserve-inference.yaml

      - name: Setup Kubectl
        uses: azure/setup-kubectl@v4

      - name: Configure Kubernetes
        shell: powershell
        run: |
            New-Item -ItemType Directory -Force "$HOME\.kube"
            '${{ secrets.KUBE_CONFIG }}' | Out-File "$HOME\.kube\config" -Encoding utf8
9
      - name: Deploy KServe
        run: |
          kubectl apply -f kserve/kserve-inference.yaml

This workflow automates the complete deployment process.

GitHub Actions Pipeline Configuration for linux machine
Add:
name: Cardio MLOps Pipeline
on:
  push:
    branches:
      - main
jobs:
  deploy:
runs-on: ubuntu-latest

steps:
- name: Checkout
      uses: actions/checkout@v4

- name: Login DockerHub
      uses: docker/login-action@v3

  with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
- name: Build Docker Image
      run: |
        docker build -t gowtham/cardio-mlops:${{ github.sha }} .
- name: Push Docker Image
      run: |
        docker push gowtham/cardio-mlops:${{ github.sha }}
- name: Replace Image Tag
      run: |
        sed -i 's|IMAGE_TAG|${{ github.sha }}|g' k8s/kserve-inference.yaml
- name: Setup Kubectl
      uses: azure/setup-kubectl@v4
- name: Configure Kubernetes
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" > ~/.kube/config
- name: Deploy KServe
      run: |
        kubectl apply -f k8s/kserve-inference.yaml

---

Understanding the Workflow Trigger
The pipeline starts when code is pushed to the main branch.
on:
  push:
    branches:
      - main
Every commit becomes a deployment candidate.
Example:
git add .
git commit -m "Updated prediction logic"
git push
Immediately triggers GitHub Actions.

---

Source Code Checkout
- name: Checkout
  uses: actions/checkout@v4
This step downloads repository contents into the GitHub Actions runner.
Without this step, the workflow has no access to the project files.

---

DockerHub Authentication
- name: Login DockerHub
Uses credentials stored in GitHub Secrets.
This allows the workflow to push images securely without exposing passwords in source code.

---

Docker Image Build
docker build -t gowtham/cardio-mlops:${{ github.sha }} .
The image tag uses:
github.sha
which represents the commit hash.
Example:
gowtham/cardio-mlops:a9f847e
Each deployment receives a unique version.

---

Docker Image Push
docker push gowtham/cardio-mlops:${{ github.sha }}
The image becomes available inside DockerHub.
Kubernetes can now pull the newly built image.

---

Updating the KServe Manifest
Inside:
k8s/kserve-inference.yaml
the image placeholder:
IMAGE_TAG
is replaced automatically.
The workflow executes:
sed -i 's|IMAGE_TAG|${{ github.sha }}|g'
Result:
image: gowtham/cardio-mlops:a9f847e
Every deployment references the correct image version.

---

Kubernetes Authentication
GitHub Actions must connect to Kubernetes.
Configuration:
echo "${{ secrets.KUBE_CONFIG }}" > ~/.kube/config
creates the kubeconfig file dynamically.
This allows:
kubectl
commands to operate against the cluster.

---

Deploying KServe
Deployment occurs with:
kubectl apply -f k8s/kserve-inference.yaml
Kubernetes detects changes and updates the running inference service.
This completes the automated deployment.

---

Managing Secrets Securely
Sensitive credentials should never be committed to Git.
GitHub provides encrypted repository secrets.
Navigate to:
GitHub Repository
     ↓
Settings
     ↓
Secrets and Variables
     ↓
Actions
Create:
DOCKER_USERNAME
DOCKER_PASSWORD
KUBE_CONFIG
These values become available during workflow execution.

---

Why Secrets Matter
Bad practice:
password: mypassword123
Good practice:
password: ${{ secrets.DOCKER_PASSWORD }}
Secrets remain encrypted and hidden from logs.

---

Verifying KServe Deployment
After deployment:
kubectl get inferenceservices

Expected:
cardio-model
Check pods:
kubectl get pods

Ensure all pods are healthy.

---

End-to-End Deployment Flow
Once CI/CD is enabled:
Developer Push
      ↓
GitHub Actions
      ↓
Docker Build
      ↓
DockerHub Push
      ↓
KServe Deployment
      ↓
Inference Service Updated
No manual deployment steps are required.

---

Final Production Flow
The complete MLOps lifecycle now becomes:
Developer Push
 ↓
GitHub Actions
 ↓
Model Training
 ↓
MLflow Tracking
 ↓
DVC Versioning
 ↓
Docker Build
 ↓
DockerHub Push
 ↓
KServe Deployment
 ↓
FastAPI Metrics
 ↓
Prometheus Scraping
 ↓
Grafana Dashboard
This represents a fully automated machine learning platform.

---

Complete Production Architecture
The final architecture is:
Browser
 ↓
Streamlit Frontend
 ↓
FastAPI Backend
 ↓
KServe
 ↓
Machine Learning Model
Monitoring Layer:
FastAPI
 ↓
Prometheus Metrics
 ↓
Prometheus
 ↓
Grafana
Deployment Layer:
GitHub
 ↓
GitHub Actions
 ↓
DockerHub
 ↓
Kubernetes
 ↓
KServe
Together these layers create a complete MLOps ecosystem.

---

Production Communication Architecture
Docker Compose
Frontend Container
 ↓
http://backend:8000
 ↓
Backend Container
The hostname:
backend
is automatically provided by Docker's internal DNS.

---

Kubernetes
Frontend Pod
 ↓
backend-service
 ↓
Backend Pod
The hostname:
backend-service
is automatically created by Kubernetes Service discovery.

---

KServe
Frontend Pod
 ↓
InferenceService Endpoint
 ↓
KServe Model Server
The frontend communicates directly with the KServe inference endpoint.

---

Enterprise Deployment Pattern
Large organizations typically separate:
Frontend
Backend
KServe
Prometheus
Grafana
into independent services.
Reasons include:
Independent Scaling
Each component scales according to demand.
Fault Isolation
Failures remain isolated.
Better Monitoring
Individual components can be monitored separately.
Easier Deployment
Services can be updated independently.
Cloud-Native Design
Aligns with Kubernetes best practices.

---

Key Production Notes
Remember:
DVC tracks datasets without storing them inside Git.
MLflow tracks experiments, parameters, and metrics.
Docker packages applications into portable containers.
DockerHub stores container images.
Minikube provides a local Kubernetes environment.
KServe automatically manages inference deployments.
Prometheus collects metrics from /metrics.
Grafana visualizes operational metrics.
GitHub Actions automates deployments.
FastAPI serves prediction APIs.
Streamlit provides a user-friendly frontend.
Kubernetes manages orchestration and scaling.

---

Conclusion
We have successfully built a complete end-to-end MLOps platform that covers the entire machine learning lifecycle:
✅ Dataset Versioning with DVC
✅ Experiment Tracking with MLflow
✅ Model Training and Evaluation
✅ FastAPI Prediction APIs
✅ Streamlit User Interface
✅ Docker Containerization
✅ DockerHub Registry Integration
✅ Kubernetes Deployment
✅ KServe Model Serving
✅ Prometheus Monitoring
✅ Grafana Visualization
✅ GitHub Actions CI/CD Automation
This architecture demonstrates how modern machine learning systems move from experimentation to production using cloud-native technologies and automated deployment pipelines.
The result is a scalable, observable, reproducible, and production-ready MLOps platform capable of supporting real-world machine learning workloads.
