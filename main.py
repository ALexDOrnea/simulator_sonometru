import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

#ignore wav warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.io.wavfile")

### VARIABILE GLOBALE
WAV_PATH="wav_samples/wavv.wav"
MODE=input("Insert mode (Fast,Slow,Peak)")
# deschidem wav
SAMPLE_RATE,AUDIO_DATA=wavfile.read(WAV_PATH)
######

#print(SAMPLE_RATE)
#print(AUDIO_DATA)

#stereo to mono(split left only)
if len(AUDIO_DATA.shape)>1:
    AUDIO_DATA=AUDIO_DATA[:,0]
    print("Audio transformat mono")
else: 
    print("Audio e deja mono")

#normalizare
if AUDIO_DATA.dtype==np.int16:
    AUDIO_NORM=AUDIO_DATA/32768.0
    print("Audio 16b normalizat")
elif AUDIO_DATA.dtype==np.int32:
    AUDIO_NORM=AUDIO_DATA/2147483648.0
    print("Audio 32b normalizat")
else:
    AUDIO_NORM=AUDIO_DATA
    print("Audio deja normalizat")
NUMBER_OF_SAMPLES=len(AUDIO_NORM)

#configurare mod
if MODE=="Fast" or MODE=="fast":
    WINDOW_SIZE=int(0.125*SAMPLE_RATE)
elif MODE=="Slow" or MODE=="slow":
    WINDOW_SIZE=int(1.0*SAMPLE_RATE)
elif MODE=="Peak" or MODE=="peak":
    WINDOW_SIZE=(0.035*SAMPLE_RATE)
# print(WINDOW_SIZE)

#Adaugam curbele A B C D



#integrare
NUMBER_OF_WINDOWS=len(AUDIO_NORM)//WINDOW_SIZE
DATE_TRUNCHIATE=AUDIO_NORM[:NUMBER_OF_WINDOWS*WINDOW_SIZE]

#transformam vectorul de date in matrice cu o fereastra pe fiecare rand
WINDOWS_MATRIX=DATE_TRUNCHIATE.reshape(NUMBER_OF_WINDOWS,WINDOW_SIZE)
#calcul valori rms direct pe vectori
RMS_VECTOR=np.sqrt(np.mean(np.square(WINDOWS_MATRIX),axis=1)) #axis decide directia de integrare(pe linii)

#epsilon foarte mic
EPSILON=1e-12
VECTOR_DB=20*np.log10(RMS_VECTOR*EPSILON)

#clip valori minime de liniste
VECTOR_DB=np.clip(VECTOR_DB,-120.0, 0.0)

print(len(VECTOR_DB))
TIME=np.arange(0,NUMBER_OF_SAMPLES)/SAMPLE_RATE
print(len(TIME))

# if MODE!="Peak" or MODE!="peak":
#     TIME=np.arange(0,NUMBER_OF_SAMPLES)/SAMPLE_RATE
#     plt.figure(figsize=(10,4))   
#     plt.plot(TIME,VECTOR_DB)   
    