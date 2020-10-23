let isRunning = false, currentAcceleration = 0, intervalId;
let btnStart = document.querySelector('#btn-start');

window.addEventListener('devicemotion', handleMotion);

function setupMotion() {
	if(!isRunning) {
		setupInterval();
	} else {
		clearInterval(intervalId);
	}
	isRunning = !isRunning;
	btnStart.textContent = isRunning ? 'Detener' : 'Iniciar';
}

function setupInterval() {
	intervalId = setInterval(() => {
		if(hasValues()) sendData(currentAcceleration);
	}, 60);
}

function hasValues() {
	return currentAcceleration.x || currentAcceleration.y || currentAcceleration.z;
}

function handleMotion(event) {
	currentAcceleration = event.accelerationIncludingGravity;
}

function sendData(acceleration) {
	if(acceleration) fetch('/accelerometer', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			x: acceleration.x,
			y: acceleration.y,
			z: acceleration.z
		}),
	}).then(r => r.json()).then(result => {
		states[result.result.toUpperCase()].setState();
	});
}