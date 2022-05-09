import cv2
import numpy as np
import datetime, time
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.torch_utils import select_device 
from utils.general import  plot_one_box
class CoinDetector():
	def __init__(self):
		self.imgsz = 640
		self.weights = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']

		self.my_confidence 		= 0.80 # 0.25
		self.my_threshold  		= 0.45 # 0.45
		self.my_filterclasses 	= None
		# my_weight					= './weights/yolov5s.pt'
		self.my_weight				= os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights","coin_v1-9_best.pt")


		set_logging()
		self.device = select_device('')
		self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
		print('>> device',self.device.type)

		# Load model
		self.model = attempt_load(self.my_weight, map_location=self.device)	# load FP32 model
		self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())	# check img_size

		if self.half:
			self.model.half()  # to FP16

		# Get names and colors
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
		self.colors = [
			(232, 182, 0), # 5Baht
			(0, 204, 255),	# 1Baht
			(69, 77, 246),	# 10Baht
			(51, 136, 222),	# 2Baht
			(222, 51, 188),	# .50Baht
			]

		# coin diameter in centimeter
		self.coin_diameter = [
			2.4, # 5Baht
			2.0,	# 1Baht
			2.6,	# 10Baht
			2.175,	# 2Baht
			1.8,	# .50Baht
		]

	def coin_prediction(self,input_img):
		img0 = input_img.copy()

		img = letterbox(img0, new_shape= self.imgsz)[0]
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img).to(self.device)
		img = img.half() if self.half else img.float()
		img /= 255.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# t1 = time_synchronized()
		pred = self.model(img, augment=True)[0]
		pred = non_max_suppression(pred,  self.my_confidence,  self.my_threshold, classes= self.my_filterclasses, agnostic=None)
		# t2 = time_synchronized()
		return img0,img,pred

	def coin_label(self,input_img):
		img0,img,pred =  self.coin_prediction(input_img)
		total = 0
		class_count = [0 for _ in range(len(self.names))]
		for i, det in enumerate(pred):
			gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
			if det is not None and len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

				for *xyxy, conf, cls in reversed(det):
					coin_size_threshold = 0.15
					coin_width = xyxy[2] - xyxy[0]
					coin_heigth = xyxy[3] - xyxy[1]
					# filter only 10 bath and coin width and heigth is lower than 0.15 % of picture
					if (coin_width/img0.shape[1] < coin_size_threshold and coin_heigth/img0.shape[0] < coin_size_threshold) :
						#mark_label
						class_count[int(cls)] += 1
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
						label = '%sbaht (%.1f%%)' % (self.names[int(cls)], conf*100)
						total += int(self.names[int(cls)])
						plot_one_box(xyxy, img0, label=label, color= self.colors[int(cls)], line_thickness=3)


		# print('Done. (%.3fs)' % (t2 - t1))
		# cv2.rectangle(img0,(0,10),(250,90),(0,0,0),-1)
		img0 = cv2.putText(img0, "10Baht "+str(class_count[2])+" coin", (10,45+25*1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
		img0 = cv2.putText(img0, " 5Baht "+str(class_count[0])+" coin", (10,45+25*2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
		img0 = cv2.putText(img0, " 2Baht "+str(class_count[3])+" coin", (10,45+25*3), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
		img0 = cv2.putText(img0, " 1Baht "+str(class_count[1])+" coin", (10,45+25*4), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
		img0 = cv2.putText(img0, " Total "+str(total)+" Baht", 			(10,45+25*5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,0), 2)
		
		return img0	

	def visualization(self,input_img):
		# return image with 10 bath detection and two point in coin
		img0,img,pred = self.coin_prediction(input_img)
		coin_point_1 = -1
		coin_point_2 = -1

		for i, det in enumerate(pred):
			gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
			if det is not None and len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

				for *xyxy, conf, cls in reversed(det):
					coin_size_threshold = 0.15
					coin_width = xyxy[2] - xyxy[0]
					coin_heigth = xyxy[3] - xyxy[1]
					# filter only 10 bath and coin width and heigth is lower than 0.15 % of picture
					if (coin_width/img0.shape[1] < coin_size_threshold and coin_heigth/img0.shape[0] < coin_size_threshold) and int(cls) == 2:
						#draw_ellipse
						center_x = int((xyxy[0]+xyxy[2])/2)
						center_y = int((xyxy[1]+xyxy[3])/2)
						a = int(coin_width/2)
						b = int(coin_heigth/2)
						theta = 0
						img0 = cv2.ellipse(img0, (center_x,center_y), (a,b),theta,0 , 360, (62, 3, 255), 2)
						coin_point_1 = [int(a* np.cos(theta)+ center_x),
										int(a* np.sin(theta)+center_y)]
						coin_point_2 = [int(-a* np.cos(theta)+ center_x),
										int(-a* np.sin(theta) + center_y)]
						
						img0 = cv2.line(img0,tuple(coin_point_1),tuple(coin_point_2),(255,0,255),2)
						img0 = cv2.circle(img0, tuple(coin_point_1), radius=7, color=(0, 255, 0), thickness=-1)
						img0 = cv2.circle(img0, tuple(coin_point_2), radius=7, color=(0, 255, 0), thickness=-1)
						img0 = cv2.putText(img0, str(self.coin_diameter[int(cls)])+" cm", (center_x-35,center_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
		return img0 
	def predict(self,input_image_bgr):
	# return list of ellipse parameter of  all detected coin  [center_x,center_y,a,b,theta]
		img0,img,pred = self.coin_prediction(input_image_bgr)
		# coin_point_1 = -1
		# coin_point_2 = -1

		result = []

		for i, det in enumerate(pred):
			gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
			if det is not None and len(det):
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

				for *xyxy, conf, cls in reversed(det):
					coin_size_threshold = 0.15
					coin_width = xyxy[2] - xyxy[0]
					coin_heigth = xyxy[3] - xyxy[1]
					# filter only 10 bath and coin width and heigth is lower than 0.15 % of picture
					if (coin_width/img0.shape[1] < coin_size_threshold and coin_heigth/img0.shape[0] < coin_size_threshold) and int(cls) == 2:
						#draw_ellipse
						center_x = int((xyxy[0]+xyxy[2])/2)
						center_y = int((xyxy[1]+xyxy[3])/2)
						a = int(coin_width/2)
						b = int(coin_heigth/2)
						theta = 0
						result.append([center_x,center_y,a,b,theta])
		return result 



if __name__ == '__main__':
	coinDetector = coin_detector()
	images = []
	input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"input")
	output_folder =os.path.join(os.path.dirname(os.path.realpath(__file__)), "output") 
	scale_percent = 50
	for filename in os.listdir(input_folder):
		img = cv2.imread(os.path.join(input_folder,filename))
		if not img is None:
			out_image = coinDetector.visualization(img)
			result = coinDetector.predict(img)
			print(result)
			show_img = cv2.resize(out_image, (int(out_image.shape[1] * scale_percent / 100), int(out_image.shape[1] * scale_percent / 100))) 
			cv2.imshow('window', show_img)
			cv2.imwrite(os.path.join(output_folder,filename), out_image)
			cv2.waitKey(0)
	cv2.destroyAllWindows()
