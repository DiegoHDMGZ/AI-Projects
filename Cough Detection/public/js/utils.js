
const txtLabel =  document.querySelector('#txtLabel');
const txtFrequency = document.querySelector('#txtFrequency');
const txtFreqAlert =  document.querySelector('#txtFreqAlert');

function setLabel(label) {
	txtLabel.textContent = label;
	txtLabel.classList.remove('tos', 'no-tos');
	if(label === 'tos') {
		txtLabel.classList.add('tos');
	} else {
		txtLabel.classList.add('no-tos');
	}
}

function setFrequency(frequency) {
	txtFrequency.textContent = Number(frequency).toFixed(4);
	if(frequency > 20 ) { 
		if (!isPlaying) {
			isPlaying = true;
			sndClick.play();
		}
		txtFreqAlert.style.display = 'block';
	} else {
		txtFreqAlert.style.display = 'none';
	}
}
