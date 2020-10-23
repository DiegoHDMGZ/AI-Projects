function clearStateColor() {
	document.body.classList.remove(
		'bg-green-400',
		'bg-yellow-400',
		'bg-red-400',
		'bg-purple-400',
		'bg-gray-200'
	)
}

function setStateColor(color) {
	clearStateColor();
	document.body.classList.add(color);
}

const states = {
	CAMINANDO: {
		setState() {
			setStateColor('bg-green-400');
			document.body.querySelector('#result').textContent = 'CAMINANDO';
		}
	},
	SALTANDO: {
		setState() {
			setStateColor('bg-yellow-400');
			document.body.querySelector('#result').textContent = 'SALTANDO';
		}
	},
	PARADO: {
		setState() {
			setStateColor('bg-red-400');
			document.body.querySelector('#result').textContent = 'PARADO';
		}
	},
	NINGUNO: {
		setState() {
			setStateColor('bg-purple-400');
			document.body.querySelector('#result').textContent = 'NINGUNO';
		}
	},
	NOT_SIGNAL: {
		setState() {
			setStateColor('bg-gray-200');
			document.body.querySelector('#result').textContent = 'SIN SEÑAL';
		}
	}
}