from urllib.request import urlopen
import numpy as np
import io
import time
import json
import socket
import requests
import math
import glob
import os
socket.setdefaulttimeout(2.0)

class Accelerometer:
	def setUp(self, url):
		self.url = url

	def readData(self):
		try:
			result = requests.get(self.url)
			accJson = result.json()
			return accJson['accel']['data']
		except Exception as ex:
			#print("ex = " , ex)
			return None
    
	def writeData(self , filename, samples):
		fileDoc = open(filename, "w")
		print("len = " , len(samples))
		maxi = 0
		mini = 1e18
		for data in samples:
			ac = data[1]
			mini = min(mini, data[0])
			maxi = max(maxi, data[0])
			jsonData = {
				"t" : data[0],
				"ax" : ac[0],
				"ay" : ac[1],
				"az" : ac[2]
			}
			json.dump(jsonData, fileDoc)
			fileDoc.write("\n")
			#fileDoc.write(str(data) + "\n")
		
		print("tiempo pasado = " , maxi - mini , " ms")
		
	

timeWindow = 5 #seconds

if __name__ == '__main__':
	
	sensor = Accelerometer()
	while (True):	
		ip = input("Ingresa la IP (ejemplo 192.168.10.5) -> ")
		urlIPWebcam = "http://" + ip + ":8080/sensors.json?sense=accel"
		sensor.setUp(urlIPWebcam)
		print("")
		print("Realizando verificacion...")
		print("")
		ok = sensor.readData()
		if ok is not None:
			print("Verificacion correcta")
			print("")
			#sensor.writeData("anterior.txt", ok)
			break
		else:
			print("ERROR. no se detectan lecturas de acelerometro")
			print("")
	
	
	print("Se iniciara la recoleccion de datos. ¡Debe mantener la accion por 5 segundos!")
	print("")
	input("Presione Enter cuando desee empezar la toma de datos\n")
	
	print("Recoleccion iniciada")
	start = time.time()
	while (True):
		t = (time.time() - start) 
		#print(t)
		if t > timeWindow:
			break
		#sensor.readData()
	
	print("Termino recoleccion")
	print("Cargando...")
	samples = sensor.readData()
	if samples is None:
		print("Hubo un error :( Corra el programa de nuevo")
	else :	
		print("")
		print("Guardando datos...")
		listTxt = glob.glob('./*.txt')
		head, curDir = os.path.split(os.getcwd())
		filename = curDir + str(len(listTxt) + 1) + ".txt"
		print("Guardando en " , filename)
		sensor.writeData(filename, samples)
		print("********************************")
		print("Se termino con exito la operacion")
