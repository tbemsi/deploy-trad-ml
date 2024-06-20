# Sklearn Model Training, Prediction & Deployment

## Overview
This project trains a machine learning model to make predictions based on provided data. 
The application is containerized using Docker. The data files are located in the `data` directory, 
with the target variable named `y`.

## Requirements
- Docker
- Python 3.11
- Flask
- Scikit-learn
- Joblib
- Pandas
- Numpy

## Setup and Usage

### Local Setup
1. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the Flask application:
    ```sh
    python app.py
    ```

### Docker Setup
1. Build the Docker image:
    ```sh
     docker build -t flask-app .   
    ```

2. Run the Docker container with a volume mapping:
    ```sh
    docker run -p 8887:8887 -v new_data:/app/new_data flask-app
    ```

### Endpoints

- **Train Model**: POST `/train`
    - Trains the model with the data in the `data` directory.
    - Request Body: `{ "model_name": "model_name" }`
    - If `model_name` is not provided, it will train all models.
    - If `data_path` is not provided, it defaults to `data`.
    - Response: `{ "message": "Model trained successfully!" }`

- **Predict**: POST `/predict`
    - Predicts the target value based on the provided features.
    - Request Body: `{ "features": [values], "model_name": "model_name" }`
    - Response: `{ "Prediction": value }`