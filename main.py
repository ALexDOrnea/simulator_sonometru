import warnings
import os
import sys
import time
import queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfilt_zi, butter, bilinear_zpk, zpk2sos 

## pentru ignorare avertismente wav
warnings.filterwarnings("ignore",category=UserWarning,module="scipy.io.wavfile")

###########################################
############Selectie in/out################
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE=True
except ImportError:
    HAS_SOUNDDEVICE=False
    print("sounddevice not found")
    sys.exit()
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("pyqtgraph not found")
    sys.exit()

input_device_id=None
output_device_id=None

if HAS_SOUNDDEVICE:
    print("\n\n~~~~~CONFIG AUDIO~~~~~")
    devices=sd.query_devices()
    print("\nInput options")
    input_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_input_channels"]>0:
            print(f"[{i}] {dev['name']} (Input channels: {dev['max_input_channels']})")
            input_indices.append(i)

    try:
        id_in=input("select device(blank for default)\n> ").strip()
        if id_in:
            id_in=int(id_in)
            if id_in in input_indices:
                input_device_id=id_in
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    print("\nOutput options")
    output_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_output_channels"]>0:
            print(f"[{i}] {dev['name']} (Output channels: {dev['max_output_channels']})")
            output_indices.append(i)

    try:
        id_out=input("select device(blank for default)\n> ").strip()
        if id_out:
            id_out=int(id_out)
            if id_out in output_indices:
                output_device_id=id_out
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    sd.default.device=(input_device_id,output_device_id)
    current_in=(
        sd.query_devices(sd.default.device[0])["name"]
        if sd.default.device[0] is not None
        else "default"
    )
    current_out=(
        sd.query_devices(sd.default.device[1])["name"]
        if sd.default.device[1] is not None
        else "default"
    )
    print(f"\n\nConfiguration:\nInput={current_in}\nOutput={current_out}\n")

###########################################
############Alte setari####################

print("~~~~~CONFIGURARE SURSA SEMNAL~~~~~")
print("1.WAV")
print("2.Live")
SURSA_OPT=input(">").strip()

WAV_PATH=None
if SURSA_OPT=="1":
    print("~~~~~CONFIGURARE PATH WAV~~~~~")
    WAV_PATH=input("path to wav file: ").strip().strip('"')
elif SURSA_OPT!="2":
    print("Invalid option")
    sys.exit()

print("~~~~~CONFIGURARE MOD~~~~~")
MODE=input("Mode(Fast, Slow, Peak)\n>  ").strip()
PEAK_MODE=MODE.strip().lower()=="peak"

# ~~~~~SELECTIE PONDERARE FRECVENTA (IEC 61672-1)~~~~~
# A-Weighting: ponderare standard pentru perceptia auditiva (majoritatea masuratorilor de zgomot)
# C-Weighting: ponderare aproape plata, folosita pentru niveluri de varf / semnale puternice
# Z-Weighting: fara ponderare (raspuns "zero"/flat), semnalul ramane nemodificat
print("\n~~~~~SELECTIE PONDERARE (WEIGHTING)~~~~~")
print("1 A-Weighting (IEC 61672)")
print("2 C-Weighting (IEC 61672)")
print("3 Z-Weighting (flat, semnal nemodificat)")
PONDERARE_OPT=input("> ").strip().lower()

if PONDERARE_OPT in("1","a","a-weighting","a-weight"):
    TIP_PONDERARE="A-Weighting"
elif PONDERARE_OPT in("2","c","c-weighting","c-weight"):
    TIP_PONDERARE="C-Weighting"
elif PONDERARE_OPT in("3","z","z-weighting","z-weight"):
    TIP_PONDERARE="Z-Weighting"
else:
    print("invalid option.using A-weighting")
    TIP_PONDERARE="A-Weighting"

# ~~~~~SELECTIE ANALIZA PE BENZI DE FRECVENTA (IEC 61260-1)~~~~~
print("\n~~~~~SELECTIE ANALIZA PE BENZI (FILTRU DE BANDA)~~~~~")
print("1 Octave intregi (1/1 octava)")
print("2 Terte de octava (1/3 octava)")
print("3 Sesimi de octava (1/6 octava)")
print("4 Doisprezecimi  (1/12 octava)")
BANDA_OPT=input("> ").strip()

FRACTIE_MAP={"1": 1,"2": 3,"3": 6,"4": 12}
if BANDA_OPT not in FRACTIE_MAP:
    print("invalid option. using 1/3")
FRACTIE_OCTAVA=FRACTIE_MAP.get(BANDA_OPT, 3)

if PEAK_MODE:
    GRAFIC_OPT=None
    print("\n~~~~~MOD PEAK~~~~~")
    print("Display db fs level")
else:
    print("\n~~~~~SELECTIE GRAFICE~~~~~")
    print("1 dB FS (level)")
    print("2 FFT (secter)")
    print("3 Benzi de octava (bars)")
    print("4 all")
    GRAFIC_OPT=input("> ").strip()
    if GRAFIC_OPT not in ("1","2","3","4"):
        print("Invalid option.using all")
        GRAFIC_OPT="4"

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE=44100
EPSILON=1e-12
AUDIO_NORM=None
data_queue=queue.Queue()
play_pointer=0

###########################################
#########CITIRE WAV SI PREGATIRE SEMNAL####

if SURSA_OPT=="1":
    if not os.path.exists(WAV_PATH):
        print(f"{WAV_PATH} not found")
        sys.exit()

    SAMPLE_RATE,AUDIO_DATA=wavfile.read(WAV_PATH)
    if AUDIO_DATA.ndim>1:
        AUDIO_DATA=AUDIO_DATA[:,0]
        print("audio transformed to mono")
    if AUDIO_DATA.dtype==np.int16:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/32768.0
    elif AUDIO_DATA.dtype==np.int32:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/2147483648.0
    elif np.issubdtype(AUDIO_DATA.dtype,np.integer):
        info=np.iinfo(AUDIO_DATA.dtype)
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/max(abs(info.min), info.max)
    else:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)

nyquist=SAMPLE_RATE/2.0

########################################################
############# FILTRE DE PONDERARE (IEC 61672) ##########
########################################################

def get_a_weighting_filter(fs):
    f1,f2,f3,f4=20.598997,107.65265,737.86223,12194.217
    A1000=-2.000
    p1,p2,p3,p4=-2*np.pi*f1,-2*np.pi*f2,-2*np.pi*f3,-2*np.pi*f4
    z=[0,0,0,0]
    p=[p1,p1,p2,p3,p4,p4]
    k=(2*np.pi*f4)**2*(10**(A1000/20))
    zeros_d,poles_d,gain_d=bilinear_zpk(z,p,k,fs)
    return zpk2sos(zeros_d,poles_d,gain_d)

def get_c_weighting_filter(fs):
    f1,f4=20.598997,12194.217
    C1000=-0.062
    p1,p4=-2*np.pi*f1,-2*np.pi*f4
    z=[0,0]
    p=[p1,p1,p4,p4]
    k=(2*np.pi*f4)**2*(10**(C1000/20))
    zeros_d,poles_d,gain_d=bilinear_zpk(z,p,k,fs)
    return zpk2sos(zeros_d,poles_d,gain_d)

if TIP_PONDERARE=="A-Weighting":
    sos_ponderare=get_a_weighting_filter(SAMPLE_RATE)
    zi_ponderare=sosfilt_zi(sos_ponderare)*0.0
elif TIP_PONDERARE=="C-Weighting":
    sos_ponderare=get_c_weighting_filter(SAMPLE_RATE)
    zi_ponderare=sosfilt_zi(sos_ponderare)*0.0
else:  # Z-Weighting: fara filtrare, raspuns flat
    sos_ponderare=None
    zi_ponderare=None

def filtreaza_block(chunk):
    """Aplica ponderarea de frecventa selectata (A, C sau Z) semnalului brut."""
    global zi_ponderare
    if sos_ponderare is None:
        return chunk.copy()
    chunk_ponderat, zi_ponderare = sosfilt(sos_ponderare, chunk, zi=zi_ponderare)
    return chunk_ponderat
