FROM python:3.8
WORKDIR /code 
COPY . .
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -e SegmentationAPI/segmentation/SeMask-FPN
RUN pip install -r requirements.txt
RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
CMD ["python", "main.py"]