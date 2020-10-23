import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.fft import fft
from sklearn.neural_network import MLPClassifier
import pickle
import random
from collections import deque 
import time

class Agent:
	def __init__(self, threshold):
		self.ax = deque([])
		self.ay = deque([])
		self.az = deque([])
		self.threshold = threshold
		
	def readData(self, x, y, z):
		self.ax.append(x)
		self.ay.append(y)
		self.az.append(z)
		if len(self.ax) > self.n:
			self.ax.popleft()
			self.ay.popleft()
			self.az.popleft()
			
	def transformSignal(self, signal, sigma):
		signalFiltered = np.copy(signal)
		signalFFT = np.array(signal, dtype = "complex_")
		signalOutput = np.array([])
		for i in range(3):
			signalFiltered[i] = gaussian_filter1d(signal[i], sigma)
			signalFFT[i] = fft(signalFiltered[i])
			signalOutput = np.concatenate((signalOutput , np.abs(signalFFT[i])))
			
		return signalOutput
	
	def loadModel(self, filename):
		self.net, self.n, self.sigma = pickle.load(open( filename, "rb" ))	
	
	def predict(self):
		ini = time.time()
		inp = self.transformSignal(np.array([np.array(self.ax), np.array(self.ay), np.array(self.az)]), self.sigma)
		inp = inp.reshape(1,-1)
		predicted = self.net.predict(inp)[0]
		prob = self.net.predict_proba(inp)
		bestProb = np.max(prob[0])
		if bestProb < self.threshold:
			return "Ninguno"
		timeElapsed = time.time() - ini
		print("timeElapsed = " , timeElapsed , " seconds")
		return predicted
				
if __name__ == '__main__':
	agent = Agent(0.8)
	agent.loadModel("model.pkl")
	while len(agent.ax) < agent.n:
		acc = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
		agent.readData(acc[0], acc[1], acc[2])
	
	for i in range(40):
		acc = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
		agent.readData(acc[0], acc[1], acc[2])
		print(agent.predict())

