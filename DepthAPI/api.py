import numpy as np
import cv2
from flask import Blueprint
from flask import request
from .depth import dpt

depth_api = Blueprint('depth_api', __name__)

@depth_api.route('/', methods=['GET'])
def main():
	return "Welcome to depth app"

@depth_api.route('/test', methods=['GET'])
def test():
	return {"test": "depth app is running"}

@depth_api.route('/inference', methods=['POST'])
def inference():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    img_bgr = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    prediction = dpt.predict(img_rgb)
    return {"prediction": prediction}