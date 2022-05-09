from flask import Blueprint
from flask import request
import cv2
import numpy as np
import os
from coin_detector import CoinDetector
import base64

coinDetector = CoinDetector()
coin_detection_api = Blueprint('coin_detection_api', __name__)

### Tests
@coin_detection_api.route('/', methods=['GET'])
def main():
	return "Welcome to coin detection app"

@coin_detection_api.route('/test', methods=['GET'])
def test():
	return {"test": "coin detection app is running"}



### Inference And Visualization
@coin_detection_api.route('/inference', methods=['POST'])
def inference():
    file = request.files["image"]
    if file.filename == "":
        return """<h1>No Image Received</h1>""", 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    img_bgr = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    prediction = coinDetector.predict(img_bgr)
    return {"prediction": prediction}



@coin_detection_api.route('/visualize', methods=['POST'])
def visualize():
    file = request.files["image"]
    if file.filename == "":
        return """<h1>No Image Received</h1>""", 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    img_bgr = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    img_bgr = coinDetector.visualization(img_bgr)
    bin = cv2.imencode('.jpg', img_bgr)[1]
    return {"visualization": str(base64.b64encode(bin),"utf-8")}




