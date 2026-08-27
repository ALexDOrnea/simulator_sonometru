import warnings 
import os
import sys
import queue 
import threading
from collections import deque
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt,sosfilt_zi,butter,bilinear_zpk,zpk2sos,lfilter

warnings.filterwarnings("ignore",category=UserWarning, module="scipy.io.wavfile")

#######INPUT SELECT

#importing libraries
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE=True
except ImportError:
    HAS_SOUNDDEVICE=False
    print("Error loading sounddevice")
    sys.exit()
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets,QtGui
except ImportError:
    print("Error loading pygtgraph")
    sys.exit()

#selection of the input device
input_device_id=None
if HAS_SOUNDDEVICE:
    print("/n/n~~~~~COUNFIG AUDIO~~~~~")
    devices=sd.query_devices()

    print("Input options:")
    input_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_input_channels"]>0:
            print(f"[{i}] {dev['name']} (Input channels: {dev['max_input_channels']})")
            input_indices.append(i)

    try:
        id_in=input("select device:\n>").strip()
        if id_in:
            id_in=int(id_in)
            if id_in in input_indices:
                input_device_id=id_in
            else:
                print("Invalid ID, using default")
    except ValueError:
        print("Invalid value,using default")

sd.default.device=(input_device_id)

current_in=(
    sd.query_devices(sd.default.device[1])["name"]
    if sd.default.device[1] is not None
    else "default"
)

print(f"\n\nInput Device={current_in}")

#####SETARI GENERALE#####

#SURSA_OPT=2

print("~~~~~CONFIG MODE")
MODE=input("Mode:\n1.Fast\n2.Slow\n3.Peak\n>").strip()
PEAK_MODE=MODE.strip().lower()=="peak"

print("~~~~~SELECT WEIGHTING MODE~~~~~")
print("1 A-Weighting (IEC 61672)")
print("2 C-Weighting (IEC 61672)")
print("3 Z-Weighting (flat, unmodified)")
PONDERARE_OPT = input("> ").strip().lower()

if PONDERARE_OPT in ("1", "a", "a-weighting", "a-weight"):
    TIP_PONDERARE = "A-Weighting"
    SIMBOL_PONDERARE = "A"
elif PONDERARE_OPT in ("2", "c", "c-weighting", "c-weight"):
    TIP_PONDERARE = "C-Weighting"
    SIMBOL_PONDERARE = "C"
elif PONDERARE_OPT in ("3", "z", "z-weighting", "z-weight"):
    TIP_PONDERARE = "Z-Weighting"
    SIMBOL_PONDERARE = "Z"
else:
    print("Invalid option. Using A-Weighting")
    TIP_PONDERARE = "A-Weighting"
    SIMBOL_PONDERARE = "A"

print("\n~~~~~Select octave band pass filter~~~~~")
print("1 Octave intregi   (1/1 octava)")
print("2 Terte de octava  (1/3 octava)")
print("3 Sesimi de octava (1/6 octava)")
print("4 Doisprezecimi    (1/12 octava)")
BANDA_OPT = input("> ").strip()

FRACTIE_MAP = {"1": 1, "2": 3, "3": 6, "4": 12}
if BANDA_OPT not in FRACTIE_MAP:
    print("Invalid Option. Using 1/3 octave")
FRACTIE_OCTAVA = FRACTIE_MAP.get(BANDA_OPT, 3)

if PEAK_MODE:
    GRAFIC_OPT = None
    print("\n~~~~~PEAK MODE~~~~~")
else:
    print("\n~~~~~GRAPH SELECT~~~~~")
    print("1 dB FS")
    print("2 FFT")
    print("3 Octave Bands")
    print("4 All")
    GRAFIC_OPT = input("> ").strip()
    if GRAFIC_OPT not in ("1", "2", "3", "4"):
        print("Invalid option.using all")
        GRAFIC_OPT = "4"

print("\n~~~~~CALIBRARE SPL~~~~~")
print("Introdu constanta de calibrare (offset in dB) pentru a converti dBFS in dB SPL.")
print("Aceasta se determina cu un calibrator acustic (ex: 94 dB SPL la 1 kHz) si")
print("reprezinta diferenta: SPL_cunoscut - dBFS_masurat.")
try:
    CALIBRARE_DB = float(input("Constanta de calibrare (dB) > ").strip().replace(",", "."))
except ValueError:
    print("Valoare invalida. se foloseste 0.0 dB (fara calibrare, ramane dBFS)")
    CALIBRARE_DB = 0.0

print(f"Constanta de calibrare folosita: {CALIBRARE_DB:+.2f} dB")

print("\n~~~~~CONFIG AUDIO OPTIMIZAT~~~~~")
BLOCKSIZE = 256
LATENCY = "low"
print("Blocksize fix: 256 samples")
print("Latency fix: low")
print(f"Blocksize folosit: {BLOCKSIZE if BLOCKSIZE else 'auto'} | Latency folosita: {LATENCY}")

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE=48000
EPSILON=1e-12
AUDIO_NORM=None

raw_queue=queue.Queue(maxsize=4)      # audio thread -> processing thread
data_queue=queue.Queue()     # processing thread -> GUI thread
stop_event=threading.Event()

#parametrul acesta controleaza doar timpul de refresh al graficelor, nu are legatura cu procesarea
GUI_REFRESH_MS=25

#benzile sunt calculate mai greu si de aceea au nevoie de un refresh rate separat
BENZI_FFT_REFRESH_HZ=20.0
BATCH_FACTOR=max(1,round((1.0/BENZI_FFT_REFRESH_HZ)*SAMPLE_RATE/BLOCKSIZE))

play_pointer=0
NIVEL_PODEA=-120.0+CALIBRARE_DB
NIVEL_PLAFON=20.0+CALIBRARE_DB

# ~~~~~CALIBRARE LIVE (actualizabila din interfata)~~~~~
# CALIBRARE_DB, NIVEL_PODEA si NIVEL_PLAFON raman globale simple, la fel ca
# inainte - marea majoritate a codului (db_din_ms, calculeaza_fft_pentru_afisare,
# calculeaza_peak_db, culoare_pentru_nivel, actualizeaza_bara etc.) le citeste
# direct, fara nicio modificare. Adaugam doar un lock si o functie de update:
# threadul de procesare citeste globalele fara lock (o simpla citire de nume e
# atomica datorita GIL-ului) - la limita intre doua click-uri pe "Aplica" poate
# aparea, cel mult, un singur bloc calculat cu o combinatie usor inconsistenta
# intre CALIBRARE_DB si NIVEL_PODEA/NIVEL_PLAFON, neglijabil pentru un afisaj
# live. Lock-ul serveste doar sa nu se suprapuna doua actualizari intre ele
# (ex. click dublu / Enter tinut apasat).
calibrare_lock=threading.Lock()

def actualizeaza_calibrare(noua_valoare):
    global CALIBRARE_DB, NIVEL_PODEA,NIVEL_PLAFON
    with calibrare_lock:
        CALIBRARE_DB=noua_valoare
        NIVEL_PODEA=-120.0+CALIBRARE_DB
        NIVEL_PLAFON=20.0+CALIBRARE_DB
        reseteaza_peak_hold()
        reseteaza_leq()

nyquist=SAMPLE_RATE/2.0

########################################################
############# FILTRE DE PONDERARE (IEC 61672) ##########
########################################################

def prewarp_frecventa(f,fs):
    """Pre-warping pentru transformarea biliniara: mapeaza fiecare frecventa de
    colt a filtrului analog prototip la o frecventa 'deformata' astfel incat,
    DUPA transformarea biliniara, raspunsul digital sa cada exact la frecventa
    tinta corecta. Fara asta, bilinear_zpk() comprima neliniar frecventele pe
    masura ce te apropii de Nyquist (fs/2) - la 48kHz, polul de 12194.217 Hz
    din A/C-weighting e la peste jumatate din Nyquist (24kHz), exact zona unde
    warping-ul devine sever (verificat: fara pre-warp, eroarea fata de curba
    IEC ideala ajungea la -15.8dB la 20kHz si -2.3dB deja la 12kHz). Cu
    pre-warp, eroarea scade sub 1dB pana la ~12-14kHz. Deasupra a ~16kHz mai
    ramane o eroare reziduala considerabila - limitare fundamentala a oricarui
    filtru IIR biliniar atat de aproape de Nyquist, nu se rezolva decat cu o
    frecventa de esantionare mai mare (ex. 96kHz)."""
    return (fs/np.pi)*np.tan(np.pi*f/fs)

def get_a_weighting_filter(fs):
    f1,f2,f3,f4=20.598997,107.65265,737.86223,12194.217
    f1,f2,f3,f4=[prewarp_frecventa(f,fs) for f in (f1,f2,f3,f4)]
    A1000=1.9997
    p1,p2,p3,p4=-2*np.pi*f1,-2*np.pi*f2,-2*np.pi*f3,-2*np.pi*f4
    z=[0,0,0,0]
    p=[p1,p1,p2,p3,p4,p4]
    k=(2*np.pi*f4)**2*(10**(A1000/20))
    zeros_d,poles_d,gain_d=bilinear_zpk(z,p,k,fs)
    return zpk2sos(zeros_d,poles_d,gain_d)

def get_c_weighting_filter(fs):
    f1,f4=20.598997,12194.217
    f1,f4=prewarp_frecventa(f1,fs),prewarp_frecventa(f4,fs)
    C1000=0.0619
    p1,p4=-2*np.pi*f1,-2*np.pi*f4
    z=[0,0]
    p=[p1,p1,p4,p4]
    k=(2*np.pi*f4)**2*(10**(C1000/20))
    zeros_d,poles_d,gain_d=bilinear_zpk(z,p,k,fs)
    return zpk2sos(zeros_d,poles_d,gain_d)

#sos_ponderare-parametrii filtrului de tip second order selection
#zi_ponderare-conditiile initiale-valoarea precedenta a filtrului
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
    """Aplica ponderarea de frecventa selectata (A, C sau Z) semnalului brut.
    Ramane in threadul AUDIO pentru ca la playback rezultatul e chiar semnalul
    care se aude - trebuie calculat 'live', in callback. E un singur filtru sos,
    deci e ieftin si nu pune presiune pe bugetul de timp real."""
    global zi_ponderare
    if sos_ponderare is None:
        return chunk.copy()
    chunk_ponderat,zi_ponderare=sosfilt(sos_ponderare,chunk,zi=zi_ponderare)
    return chunk_ponderat

########################################################
##### CORECTIE CURBA MICROFON (compensare raspuns) ######
########################################################
# Filtru IIR (cascada biquad-uri EQ parametric, formule RBJ) care aplica
# INVERSUL curbei de calibrare individuale a microfonului, astfel incat
# raspunsul final sa fie cat mai aproape de plat. Se aplica DOAR pe semnalul
# de la microfon (SURSA_OPT=="2"), niciodata pe redarea unui WAV - acolo nu
# exista microfon in lant, deci nimic de corectat.
# Fisier ales: 30degree (montaj cu incidenta oblica, nu axiala - vezi poza
# cu suportul dublu microfon-test/B&K, unghi ~20-30 grade fata de difuzor).

CALE_CALIBRARE_MICROFON="microphone_profiles/35Y228_cal_0degree.txt"
# CALE_CALIBRARE_MICROFON="35Y228_cal_0degree.txt"

def incarca_curba_calibrare(path):
    date=np.loadtxt(path)
    return date[:,0],date[:,1]

def construieste_filtru_corectie(fs,freq_cal,dev_cal_db,numtaps=513):
    """FIR proiectat prin frequency-sampling (firwin2), care reproduce DIRECT
    inversul curbei de calibrare, fara acumulare de gain din filtre suprapuse
    (spre deosebire de cascada de biquad-uri). Adauga latenta ~ (numtaps-1)/(2*fs)
    - la 513 taps si 48kHz, ~5.3ms, comparabil cu BLOCKSIZE-ul deja folosit."""
    freqs_norm=np.concatenate(([0.0],freq_cal/(fs/2.0),[1.0]))
    freqs_norm=np.clip(freqs_norm,0.0,1.0)
    # eliminam duplicate/neordonate care ar bloca firwin2
    freqs_norm,idx_unic=np.unique(freqs_norm,return_index=True)

    gain_liniar=10**(-dev_cal_db/20.0)
    gain_extins=np.concatenate(([gain_liniar[0]],gain_liniar,[gain_liniar[-1]]))
    gain_extins=gain_extins[idx_unic]

    from scipy.signal import firwin2
    taps=firwin2(numtaps,freqs_norm,gain_extins)
    return taps

sos_corectie_mic = None
zi_corectie_mic = None

try:
    freq_cal,dev_cal_db=incarca_curba_calibrare(CALE_CALIBRARE_MICROFON)
    sos_corectie_mic=construieste_filtru_corectie(SAMPLE_RATE,freq_cal,dev_cal_db)
    if sos_corectie_mic is not None:
        zi_corectie_mic=sosfilt_zi(sos_corectie_mic)*0.0
        print(f"mic corection loaded {CALE_CALIBRARE_MICROFON}"
              f"({len(sos_corectie_mic)} actice EQ bands)")
    else:
        print("Curba de calibrare nu a generat nicio corectie (deviatii neglijabile).")
except (OSError, ValueError) as e:
    print(f"calibration curve not loaded:{e}")s
    sos_corectie_mic=None

