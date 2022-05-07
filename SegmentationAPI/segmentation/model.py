import os
import mmcv
from mmcv import Config
from mmseg.apis import init_segmentor

config = os.path.abspath('SegmentationAPI/segmentation/configs/configs.py')
checkpoint = os.path.abspath('SegmentationAPI/segmentation/checkpoints/final.pth')
model = init_segmentor(config, checkpoint)

