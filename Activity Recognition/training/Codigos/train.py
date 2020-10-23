import numpy as np
import time
import os
import json
from scipy.ndimage.filters import gaussian_filter1d
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#from sklearn.externals import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import cmath

class Agent:
	def getFilesInDirectory(self, directory):
		files = []
		for dirpath,_,filenames in os.walk(directory):
			for f in filenames:
				files.append(os.path.abspath(os.path.join(dirpath, f)))
		return files

	def getClass(self, path):
		head, tail = os.path.split(path)
		head , tail = os.path.split(head)
		return tail
	
	def getPerson(self, path):
		head, tail = os.path.split(path)
		head , tail = os.path.split(head)
		head , tail = os.path.split(head)
		return tail
		
	def readData(self, pathDir):
		pathTrain = pathDir + "/train"
		pathTest = pathDir + "/test"
		self.filesTrain = self.getFilesInDirectory(pathTrain)
		self.filesTest = self.getFilesInDirectory(pathTest)
		
	def getSignal(self, filename):
		fileDoc = open(filename, "r")
		ax = []
		ay = []
		az = []
		for dataString in fileDoc:
			dataJson = json.loads(dataString)
			ax.append(dataJson['ax'])
			ay.append(dataJson['ay'])
			az.append(dataJson['az'])
			
		return np.array([np.array(ax), np.array(ay) , np.array(az)])
	
	def transformSignal(self, signal, sigma):
		signalFiltered = np.copy(signal)
		signalFFT = np.array(signal, dtype = "complex_")
		signalOutput = np.array([])
		for i in range(3):
			signalFiltered[i] = gaussian_filter1d(signal[i], sigma)
			signalFFT[i] = fft(signalFiltered[i])
			
			#signalOutput = np.concatenate((signalOutput , signalFFT[i].real, signalFFT[i].imag))
			signalOutput = np.concatenate((signalOutput , np.abs(signalFFT[i])))
			#signalOutput = np.concatenate((signalOutput , np.angle(signalFFT[i])))
			#signalOutput = np.concatenate((signalOutput, signalFiltered[i]))
			
		
		"""fig, (ax, ay) = plt.subplots(nrows=2, ncols=1)
		ax.plot(signalFiltered[0])
		ax.title.set_text("Señal dominio de tiempo")
		
		n = 15
		T = 1 / 15
		f = np.linspace(0.0, 1.0/T, n)
		
		print("f = " , f)
		print("signal = " , np.abs(signalFFT[0]))
		ay.plot(f , np.abs(signalFFT[0]))
		ay.title.set_text("Señal dominio frecuencia")
		plt.tight_layout()
		plt.show()"""
		return signalOutput
		
	
	def preprocess(self, sigma):
		self.trainingSignals = []
		self.trainingLabels = []
		
		self.testSignals = []
		self.testLabels = []
		for curFile in self.filesTrain:
			#if self.getClass(curFile) == 'caminando':
			signal = self.transformSignal(self.getSignal(curFile), sigma)
			self.trainingSignals.append(signal)
			self.trainingLabels.append(self.getClass(curFile))
			
		
		for curFile in self.filesTest:
			signal = self.transformSignal(self.getSignal(curFile), sigma)
			#if self.getPerson(curFile) != 'Persona2':
			self.testSignals.append(signal)
			self.testLabels.append(self.getClass(curFile))
				#print(self.getPerson(curFile))
		
		self.scaler = None
		#self.scaler = StandardScaler()
		#self.trainingSignals = self.scaler.fit_transform(self.trainingSignals)
		#self.testSignals = self.scaler.transform(self.testSignals)
	
	def train(self, hidden_layers, lr, epochs, alpha, pathDesc):
		self.net = MLPClassifier(hidden_layer_sizes = hidden_layers, learning_rate_init= lr, max_iter= epochs, tol = 1e-5, alpha = alpha)
		#self.net = MLPClassifier()
		self.net = self.net.fit(self.trainingSignals, self.trainingLabels)
		fileDoc = open(pathDesc + "/descripcion.txt", "w")
		fileDoc.write("hidden_layer_sizes = " + str(hidden_layers) + "\n")
		fileDoc.write("learning rate = "  + str(lr) + "\n")
		fileDoc.write("alpha = "  + str(alpha) + "\n")
		if self.scaler is not None:
			fileDoc.write("with scaler\n")
		else :
			fileDoc.write("no scaler\n")
	
	def test(self, pathMetrics):
		accuracy = self.net.score(self.trainingSignals, self.trainingLabels)
		print("accuracy training = " , accuracy)
		fileDoc = open(pathMetrics + "/accuracy.txt", "w")
		fileDoc.write("accuracy training = " + str(accuracy) + "\n")
		
		accuracy = self.net.score(self.testSignals, self.testLabels)
		print("accuracy test = " , accuracy)
		fileDoc.write("accuracy test = " + str(accuracy) + "\n")
	
	def saveModel(self, filename):
		#joblib.dump(self.net, filename)
		pickle.dump(self.net, open( filename, "wb" ))
		#pickle.dump((self.net, self.scaler), open( filename, "wb" ))
			
			
			
				
if __name__ == '__main__':
	agent = Agent()
	start = time.time()
	pathDir = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/dataset"
	print("reading...")
	agent.readData(pathDir)
	print("end reading")
	print("")
	
	print("preprocessing...")
	agent.preprocess(2)
	print("end preprocessing")
	print("")
	
	hidden_layers = (200, 100)
	learning_rate = 1e-4
	epochs = 400
	alpha = 1e-5
	print("training...")
	pathMetrics = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/model"
	agent.train(hidden_layers , learning_rate, epochs, alpha, pathMetrics)
	
	print("end training")
	print("")
	
	
	agent.test(pathMetrics)
	
	print("saving model...")
	filename = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/model/model.pkl"
	agent.saveModel(filename)
	print("model saved")
	

