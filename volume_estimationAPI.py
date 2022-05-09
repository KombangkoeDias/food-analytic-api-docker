from flask import Blueprint
from flask import request
import cv2
import numpy as np
import os
from volume_estimator import VolumeEstimator

volumeEstimator = VolumeEstimator()


volume_estimation_api = Blueprint('volume_estimation_api', __name__)

### Tests
@volume_estimation_api.route('/', methods=['GET'])
def main():
	return "Welcome to volume estimation app"

@volume_estimation_api.route('/test', methods=['GET'])
def test():
	return {"test": "volume estimation app is running"}



### Inference And Visualization
@volume_estimation_api.route('/inference', methods=['POST'])
def inference():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    return volume_estimation_inference(img_numpy)


@volume_estimation_api.route('/totalVolume', methods=['POST'])
def totalVolume():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    return {"prediction":volume_estimation_inference(img_numpy)["prediction"]["total"]}


# @volume_estimation_api.route('/visualize', methods=['POST'])
# def visualize():
#     prediction = inference()['prediction']
#     base64Image = SeMask_FPN.visualization.getVisualization(prediction, base64=True) # visualization
#     return {"prediction": prediction, "visualization": base64Image}


def volume_estimation_inference(img):
    imgBGR = cv2.imdecode(img, cv2.IMREAD_COLOR)
    prediction = volumeEstimator.estimate_volume(imgBGR) # prediction
    return {"prediction": prediction}

