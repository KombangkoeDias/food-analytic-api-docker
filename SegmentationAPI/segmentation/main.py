import os
from .visualization import SegmentationVisualization
import torch
from mmseg.apis import inference_segmentor
from .model import model
from .utils import config
import numpy as np
import cv2

class SegmentationInferenceWrapper():
    def __init__(self, model):
        self.model = model
        self.model.config = config
        self.visualization = SegmentationVisualization(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, img):
        pass

class SeMask_FPN_InferenceWrapper(SegmentationInferenceWrapper):
    def __init__(self, model):
        super().__init__(model)    

    # override
    def predict(self, img):
        prediction = inference_segmentor(self.model, img)
        prediction = np.array(prediction[0])[0].tolist()
        return prediction

SeMask_FPN = SeMask_FPN_InferenceWrapper(model)




