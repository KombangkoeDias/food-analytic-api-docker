import os

dpt_config = {
    'scale': 0.0000305,
    'shift': 0.1378,
    'image_size': (384, 384),
    'model_path': os.path.abspath('DepthAPI/depth/weights/dpt.pt'),
}