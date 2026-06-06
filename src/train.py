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

