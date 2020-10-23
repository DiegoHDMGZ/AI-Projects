import numpy as np
from fastai.vision import *
import fastai
from PIL import Image as PImage
import matplotlib.pyplot as plt
import time
import ctypes
import torchvision.transforms as T
import io
import base64
import os
#ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

class Model:
	def warmup(self):
		arr = np.random.randint(0,255,(100,100,3))
		pil_image = PImage.fromarray(arr,'RGB')
		tensor = img_tensor = T.ToTensor()(pil_image)
		img = Image(tensor)
		self.net.predict(img)
	
	def loadModel(self):
		self.net = load_learner(os.getcwd(), file='model.pkl')
		self.warmup()

class Agent:
	def __init__(self):
		self.t_inicial = None
		self.duracion_tos = 0.65
		self.activo = False
	
	def loadModel(self, model):
		self.net = model.net
		
	def readData(self, img_encoded):
		image_bytes = io.BytesIO(base64.b64decode(img_encoded))
		self.pil_image = PImage.open(image_bytes)
		self.pil_image = self.pil_image.convert('LA').convert('RGB')
		tensor = img_tensor = T.ToTensor()(self.pil_image)
		self.img = Image(tensor)
	
	def updateFrequency(self, label):
		if label != 'tos':
				if self.activo:
					self.t_total_tos += time.time() - self.t_inicial_tos 
					self.activo = False
		elif not self.activo:
				self.activo = True
				self.t_inicial_tos = time.time()
	def predict(self):
		if self.t_inicial is None:
			self.t_inicial = time.time()
			self.t_total_tos = 0
		try:
			pred = self.net.predict(self.img)
			label = str(pred[0])
			self.updateFrequency(label)
			return label 
		except Exception as ex:
			return None
	
	def getFrequency(self):
		cur_t_tos = self.t_total_tos
		if self.activo:
			cur_t_tos += time.time() - self.t_inicial_tos 
		
		num_tosidos =  cur_t_tos / self.duracion_tos
		t_transcurrido = (time.time() - self.t_inicial) / 60
		frecuencia = num_tosidos / t_transcurrido
		return frecuencia
	
