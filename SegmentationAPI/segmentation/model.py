import os
from mmcv import Config
from mmseg.apis import init_segmentor

config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './configs/SETR_MLA_768x768_80k_base_on_FoodSeg73.py')
checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), './checkpoints/iter_80000.pth')
model = init_segmentor(config, checkpoint)

