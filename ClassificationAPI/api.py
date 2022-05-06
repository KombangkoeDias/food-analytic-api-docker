from flask import request, jsonify, Blueprint
from PIL import Image
import torch
from torch import nn
from .utils import transform
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
        
    prediction = []
    for i in range(config["top_k"]):
        classname = config["idx_to_class"][topKPredictedClassIdx[i].item()]
        prob = topKPredictedClassValue[i].item()
        prediction.append([classname, prob])

    return jsonify({"prediction": prediction})