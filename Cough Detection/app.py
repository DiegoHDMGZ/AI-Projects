from flask import Flask, send_from_directory, request
from typing import BinaryIO
from predict import Agent, Model
import time

app = Flask(__name__)

model = Model()
model.loadModel()
agents = {}

PUBLIC_PATH = './public'

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
  return send_from_directory(PUBLIC_PATH, path)

@app.route('/')
def root():
  return send_from_directory(PUBLIC_PATH, 'index.htm')

@app.route('/result')
def result():
  return send_from_directory(PUBLIC_PATH, 'result.html')

@app.route('/upload-photo', methods=['POST'])
def on_photo_upload():
	ip = None
	try:
		ip = request.headers['forwarded'].split('=')[1]
		print("ip: " + ip)
		print("forwarded: " + request.headers['forwarded'])
	except Exception:
		ip = 'localhost'
	agent = agents.get(ip)
	if agent == None:
		agent = Agent()
		agent.loadModel(model)
		agents[ip] = agent
	agent.readData(request.json['photo'])
	# Retorna la predicción
	# success: False si ocurren errores, result tiene el resultado de la predicción
	label = agent.predict()
	frecuencia = agent.getFrequency()
	result = {
		'label': label,
		'frecuencia': frecuencia
	}
	return {'success': True,  'result': result}


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000, debug=True)
