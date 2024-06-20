from flask import Flask, request, jsonify
import numpy as np
from .model import MachineLearningModel


app = Flask(__name__)
model = MachineLearningModel(features=[f"feature_{i}" for i in range(6)])


@app.route("/train", methods=["POST"])
def train():
    model_name = request.json.get("model_name")
    data_path = request.json.get("data_path", "data")

    model.train(model_name=model_name, data_path=data_path)
    if model_name:
        message = jsonify({"message": f"Model {model_name} trained successfully"})
    else:
        message = jsonify({"message": "All models trained successfully"})
    return message


@app.route("/predict", methods=["POST"])
def predict():
    datapoints = np.array(request.json["datapoints"]).reshape(1, -1)
    model_name = request.json.get("model_name")
    if not model_name:
        return jsonify({"error": "Model name is required for prediction"}), 400
    prediction = model.predict(datapoints, model_name=model_name)
    return jsonify({"Prediction": prediction[0]})


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Server is running!"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8887)
