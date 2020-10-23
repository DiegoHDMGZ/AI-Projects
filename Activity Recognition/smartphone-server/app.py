from flask import Flask, send_from_directory, request
from predict import Agent

app = Flask(__name__)
agent = Agent(0.8)
agent.loadModel("model.pkl")

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
  return send_from_directory('./dist', path)

@app.route('/')
def root():
  return send_from_directory('./dist', 'index.html')

@app.route('/accelerometer', methods=['POST'])
def on_accelerometer_input():
	x, y, z = request.json.values() # Haz lo que quieras con x, y, z
	agent.readData(x, y, z)
	# Retorna la predicción 
	# success: False si ocurren errores, result tiene el resultado de la predicción
	return { 'success': True,  'result': agent.predict() }

if __name__ == "__main__":
	app.run(host = '0.0.0.0', port=8080, debug=True)
