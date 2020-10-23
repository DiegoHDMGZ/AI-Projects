import numpy as np
import time
import os
import json
from scipy.ndimage.filters import gaussian_filter1d
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import itertools
import csv

import warnings
warnings.filterwarnings("ignore")

class Agent:
	def plotConfusionMatrix(self, cm,target_names,title='Confusion matrix',cmap=None,normalize=True, nameFile = None):
		accuracy = np.trace(cm) / float(np.sum(cm))
		misclass = 1 - accuracy

		if cmap is None:
			cmap = plt.get_cmap('Blues')

		plt.figure(figsize=(8, 10))
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()

		if target_names is not None:
			tick_marks = np.arange(len(target_names))
			plt.xticks(tick_marks, target_names, rotation=45)
			plt.yticks(tick_marks, target_names)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


		thresh = cm.max() / 1.5 if normalize else cm.max() / 2
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if normalize:
				plt.text(j, i, "{:0.4f}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="black" if cm[i, j] > thresh else "black")
			else:
				plt.text(j, i, "{:,}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="black" if cm[i, j] > thresh else "black")


		plt.tight_layout()
		plt.xlabel('True label\naccuracy={:0.4f}; error={:0.4f}'.format(accuracy, misclass))
		plt.ylabel('Predicted label')
		
		if nameFile is not None:
			plt.savefig(nameFile)
		plt.close()
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
			signalOutput = np.concatenate((signalOutput , np.abs(signalFFT[i])))
			
		return signalOutput
		
	
	def preprocess(self, sigma):
		self.sigma = sigma
		self.trainingSignals = []
		self.trainingLabels = []
		
		self.testSignals = []
		self.testLabels = []
		for curFile in self.filesTrain:
			signal = self.transformSignal(self.getSignal(curFile), sigma)
			self.trainingSignals.append(signal)
			self.trainingLabels.append(self.getClass(curFile))
			
		
		for curFile in self.filesTest:
			signal = self.transformSignal(self.getSignal(curFile), sigma)
			self.testSignals.append(signal)
			self.testLabels.append(self.getClass(curFile))
		
	
	def train(self, hidden_layers, lr, epochs, alpha, pathDesc):
		self.net = MLPClassifier(hidden_layer_sizes = hidden_layers, learning_rate_init= lr, max_iter= epochs, tol = 1e-5, alpha = alpha, early_stopping = False , random_state = 42)
		self.n = self.trainingSignals[0].shape[0] // 3
		self.net = self.net.fit(self.trainingSignals, self.trainingLabels)
		fileDoc = open(pathDesc + "/descripcion.txt", "w")
		fileDoc.write("hidden_layer_sizes = " + str(hidden_layers) + "\n")
		fileDoc.write("learning rate = "  + str(lr) + "\n")
		fileDoc.write("alpha = "  + str(alpha) + "\n")
		fileDoc.write("epochs = "  + str(epochs) + "\n")
	
	def test(self, pathMetrics):
		accuracy = self.net.score(self.trainingSignals, self.trainingLabels)
		#print("accuracy training = " , accuracy)
		fileDoc = open(pathMetrics + "/accuracy.txt", "w")
		fileDoc.write("accuracy training = " + str(accuracy) + "\n")
		
		accuracy = self.net.score(self.testSignals, self.testLabels)
		
		target_names = ['parado' ,'caminando', 'saltando']
		prediction = self.net.predict(self.testSignals)
		cm = confusion_matrix(self.testLabels , prediction, target_names)
		self.plotConfusionMatrix(cm, target_names, normalize = False, nameFile = pathMetrics + "/confusion.jpg")
		fileDoc.write("accuracy test = " + str(accuracy) + "\n")
		return accuracy
	
	def saveModel(self, path):
		pickle.dump((self.net, self.n, self.sigma), open( path + "/model.pkl", "wb" ))
			

def gridSearch(hidden_layers_grid, lr_grid, epochs_grid, alpha_grid):
	agent = Agent()
	pathDir = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/Prueba 2/dataset"
	print("reading...")
	agent.readData(pathDir)
	print("end reading")
	print("")
	
	print("preprocessing...")
	agent.preprocess(2)
	print("end preprocessing")
	print("")
	
	pathSave = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/Prueba 2/model"
	cnt = 1
	
	fileWrite = open(pathSave + "/grid_search.csv", "w",  newline='')
	writer = csv.writer(fileWrite)
	writer.writerow(["model" , "capas" ,"nodos" , "learning rate" , "epochs" , "alpha", "accuracy"])
	bestAccuracy = 0
	bestModel = -1
	for hidden_layers in hidden_layers_grid:
		for learning_rate in lr_grid:
			for epochs in epochs_grid:
				for alpha in alpha_grid:
					path = pathSave + "/model" + str(cnt)
					os.mkdir(path)
					agent.train(hidden_layers , learning_rate, epochs, alpha, path)
					accuracy = agent.test(path)
					agent.saveModel(path)
					layers = [agent.n * 3]
					if type(hidden_layers) == tuple:
						for l in hidden_layers:
							layers.append(l)
					else:
						layers.append(hidden_layers)
					layers.append(3)
					row = np.array([str(cnt), str(len(layers)) , str(layers) , str(learning_rate), str(epochs), str(alpha), str(accuracy)])
					writer.writerow(row)
					
					if accuracy > bestAccuracy:
						bestAccuracy = accuracy
						bestModel = cnt
					
					cnt += 1
	fileWrite.close()
	print("best model = model" , bestModel)
	print("accuracy = " , bestAccuracy)
			
				
if __name__ == '__main__':
	gridSearch([(10, 5) , (30), (100, 30), (200, 100), (80, 40, 10)], [1e-2, 1e-3, 1e-4], [200, 300 , 400], [1e-5, 1e-2])
	"""agent = Agent()
	pathDir = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/dataset"
	print("reading...")
	agent.readData(pathDir)
	print("end reading")
	print("")
	
	print("preprocessing...")
	agent.preprocess(2)
	print("end preprocessing")
	print("")
	
	hidden_layers = (10, 5)
	learning_rate = 1e-3
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
	print("model saved")"""
	

