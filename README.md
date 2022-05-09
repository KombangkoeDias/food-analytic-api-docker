# food-image-analytic-API

## ClassificationAPI
- Download [model weight](https://drive.google.com/file/d/1--rIEfBR6AgPXFXzldA30EdA9bk_Y_j6/view?usp=sharing), place the file in /ClassificationAPI, and rename to "model_weight.pt"
- Optional: you can also change the value (k) of "top_k" in /ClassificationAPI/config.py to any number from 1 to 330, this will return the top-k class prediction.

## SegmentationAPI
- Installation : do installation steps according to SegmentationAPI/install_steps.sh
- segmentation api function (named segmentation_inference) can be imported from the SegmentationAPI module if needed. This function will take an input image and provide prediction.
