from flask import Blueprint
from flask import request
import cv2
import numpy as np
import os
from .segmentation import utils
from .segmentation import SeMask_FPN

segmentation_api = Blueprint('segmentation_api', __name__)

### Tests
@segmentation_api.route('/', methods=['GET'])
def main():
	return "Welcome to segmentation app"

@segmentation_api.route('/test', methods=['GET'])
def test():
	return {"test": "segmentation app is running"}

@segmentation_api.route('/test/inference', methods=['GET'])
def test_inference():
    return {"test_inference":  SeMask_FPN.predict(os.path.abspath('imgs/test.jpg'))}

### Classes
@segmentation_api.route('/classes/id2label', methods=['GET'])
def get_id2label():
    return {'id2label': utils.id2label}

@segmentation_api.route('/classes/label2id', methods=['GET'])
def get_label2id():
    return {'label2id': utils.label2id}

### Inference And Visualization
@segmentation_api.route('/inference', methods=['POST'])
def inference():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    imgBGR = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    return segmentation_inference(imgBGR)

def segmentation_inference(imgBGR):
    prediction = SeMask_FPN.predict(imgBGR) # prediction
    base64Image = SeMask_FPN.visualization.getVisualization(prediction, base64=True) # visualization
    return {"prediction": prediction, "visualization": base64Image}



