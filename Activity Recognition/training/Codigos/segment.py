import numpy as np
import time
import os

class Agent:
	def getFilesInDirectory(self, directory):
		files = []
		for dirpath,_,filenames in os.walk(directory):
			for f in filenames:
				files.append(os.path.abspath(os.path.join(dirpath, f)))
		return files

	def getParts(self, path):
		head, filename = os.path.split(path)
		head , curClass = os.path.split(head)
		head, person = os.path.split(head)
		return filename, curClass, person
	

	def readData(self, pathDir):
		self.files = self.getFilesInDirectory(pathDir)
	
	def segment(self, signal, numSamples, numOverlap):
		segmentedSignal = []

		for i in range(0, len(signal) , numSamples - numOverlap):
			if i + numSamples > len(signal):
				break
			segmentedSignal.append(signal[slice(i , i + numSamples, 1)])
	
		return segmentedSignal
		
	def saveSignal(self, signal, filename):
		fileDoc = open(filename, "w")
		for data in signal:
			fileDoc.write(data)
	
	def writeData(self, numSamples, numOverlap, pathOutput):
		cnt = 0
		for curFile in self.files:
			filename, curClass, person = self.getParts(curFile)
			fileDoc = open(curFile, "r")
			signal = []
			for data in fileDoc:
				signal.append(data)
			
			segmentedSignal = self.segment(signal, numSamples, numOverlap)
			for signal in segmentedSignal:
				cnt += 1
				path = pathOutput + "/" + person + "/" + curClass + "/activity" + str(cnt).zfill(4) + ".txt"
				self.saveSignal(signal, path)

if __name__ == '__main__':
	agent = Agent()
	pathData = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/Prueba 2/rawDataset"
	agent.readData(pathData)
	pathOutput = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/Prueba 2/dataset"
	agent.writeData(15, 3, pathOutput)
	


