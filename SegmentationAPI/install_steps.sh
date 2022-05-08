conda create -n foodapi python=3.8
conda activate foodapi
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim==0.1.5
mim install mmcv-full==1.5.0
pip install terminaltables==3.1.10 timm==0.5.4 mmcls==0.23.0
pip install pillow requests numpy tqdm pandas
pip install -e SegmentationAPI/segmentation/SeMask-FPN
pip install -r requirements.txt
pip install gdown
# download the checkpoint file.
gdown --id 141lq_mb6Ayku4u5abMoWgcpvhLWDrxfw -O SegmentationAPI/segmentation/checkpoints/final.pth
