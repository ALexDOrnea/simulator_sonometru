# simulator_sonometru


Accepta audio live sau fisiere wav si afiseaza grafice pentru dB si fft cu si fara filtru lowpass/highpass + afiseaza la final spectrele finale de
amplitudine

Fisiere:
* main.py --varinata originala cu lowpass si highpass
* ponderi.py --varianta 2 cu implementare pentru ponderile A si C
* optimized_test.py --varianta cea mai noua cu grafice mai smechere

Dependente:
* matploit
* numpy
* scipy
* sounddevice
* pyqtgraph
* pyqt5
* pyinstaller
* snakeviz (optional)

Tasks
* verificare mod mediere fereastra -- parca am facut
* despartire cod in 2 threaduri: thread de obtinere date audio si thread de procesare -- facut
* verificare caracteristica diferite microfoane
* optimizare gui -- facut partial

test