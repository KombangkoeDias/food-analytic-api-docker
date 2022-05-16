FROM python:3.8
WORKDIR /code 
COPY requirements.txt . 
RUN pip install -r requirements.txt
RUN pip install -e SegmentationAPI/segmentation/SeMask-FPN
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
COPY . .
CMD ["python", "./main.py"]