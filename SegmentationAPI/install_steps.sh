conda create -n foodapi python=3.8
conda activate foodapi
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full
pip install terminaltables timm mmcls
pip install pillow requests numpy tqdm pandas
pip install -e SegmentationAPI/segmentation/SeMask-FPN
pip install -r requirements.txt
pip install gdown
# download the checkpoint file.
gdown --id 1-9YxLCHIFV78BjVATlIA0DIIKRSabbwW -O SegmentationAPI/segmentation/checkpoints/final.pth
