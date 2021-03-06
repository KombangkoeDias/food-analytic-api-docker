from flask import request, jsonify, Blueprint
from PIL import Image
import torch
from torch import nn
from .utils import transform, find_nutrients_dict
from .model import model
from .config import config


classification_api = Blueprint('classification_api', __name__)

@classification_api.route('/', methods=['GET'])
def main():
    return '''<h1>Welcome to classification app</h1>'''

@classification_api.route('/inference', methods=['POST'])
def receiveFoodImageAndClassify():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    image = Image.open(file)
    image_tensor = transform(image)
    with torch.no_grad():
        output = model(torch.unsqueeze(image_tensor,0))[0]  #torch.Tensor 1D
        output = nn.Softmax(dim=0)(output)
        topKPredictedClassValue, topKPredictedClassIdx = torch.topk(output, config["top_k"])
        
    predictions = []
    for i in range(config["top_k"]):
        prediction = {}
        prediction["food name"] = config["idx_to_class"][topKPredictedClassIdx[i].item()]
        prediction["probability"] = topKPredictedClassValue[i].item()
        prediction["nutrients (per 100g of food)"] = find_nutrients_dict(prediction["food name"])
        predictions.append(prediction)

    return jsonify({"predictions": predictions})