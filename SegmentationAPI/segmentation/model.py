import os
import mmcv
from mmcv import Config
from mmseg.apis import init_segmentor
import torch

config = os.path.abspath('SegmentationAPI/segmentation/configs/fpn_semask_base_fp16_640x640_800k_foodseg103.py')
checkpoint = os.path.abspath('SegmentationAPI/segmentation/checkpoints/final.pth')

if torch.cuda.is_available():
    model = init_segmentor(config, checkpoint)
else:
    model = init_segmentor(config, checkpoint, 'cpu')

