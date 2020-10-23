const socket = io('/');

socket.on('connect', () => {
	console.log('Connected!');
});

socket.on('result', (result) => {
	if(result.label) {
		setLabel(result.label);
		setFrequency(result.frecuencia);
	}
})