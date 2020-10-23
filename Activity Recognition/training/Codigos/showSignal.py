import numpy as np
import time
import os
import matplotlib.pyplot as plt
import json

def getFilesInDirectory(directory):
	files = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			files.append(os.path.abspath(os.path.join(dirpath, f)))
	return files

def getParts(path):
	head, tail = os.path.split(path)
	head , activity = os.path.split(head)
	head, person = os.path.split(head)
	return activity, person

def getSignal(filename):
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

path = "D:/Diego/UNI/9no ciclo/IA/examen-final/training/Prueba 2/rawDataset"
files = getFilesInDirectory(path)

descript = {}
descript['Persona1'] = 'Hombre joven'
descript['Persona2'] = 'Hombre adulto'
descript['Persona3'] = 'Mujer joven'
descript['Persona4'] = 'Mujer adulta'

sampleSignal = {'caminando' : {} , 'saltando' : {} , 'parado' : {}}
for i in range(len(files) - 1 , -1, -1):
	curFile = files[i]
	signal = getSignal(curFile)
	activity, person = getParts(curFile)
	sampleSignal[activity][person] = signal




for activity in sampleSignal:
	fig, ax = plt.subplots(nrows=4, ncols=1)
	i = 0
	for person in sampleSignal[activity]:
		signal = sampleSignal[activity][person]
		ax[i].title.set_text(descript[person] + " - " + activity)
		for k in range(3):
			ax[i].plot(signal[k])
		
		
		i += 1
	
	plt.tight_layout()
	plt.show()



