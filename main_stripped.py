import warnings
import os
import sys
import queue
import threading
from collections import deque
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt,sosfilt_zi,butter,bilinear_zpk,zpk2sos,lfilter

warnings.filterwarnings("ignore",category=UserWarning,module="scipy.io.wavfile")

#input select
#sd
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE=True
except ImportError:
    HAS_SOUNDDEVICE=False
    print("Error initialising sounddevice")
    sys.exit()
#pygraph
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore,QtWidgets,QtGui
except ImportError:
    print("Error loading pyqtgraph")
    sys.exit()

#input select
input_device_id=None
if HAS_SOUNDDEVICE:
    print("/n/nInput device select")
    devices=sd.query_devices()
    print("Options:")
    input_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_input_channels"]>0:
            print(f"[{i}] {dev['name']} (Input channels: {dev["max_input_channels"]})")
            input_indices.append(i)
    try:
        id_in=input("select device:\n>").strip()
        if id_in:
            id_in=int(id_in)
            if id_in in input_indices:
                input_device_id=id_in
            else:
                print("invalid id, using default device")
    except ValueError:
        print("Invalid value,using default")

sd.default.device=(input_device_id)

current_in=(
    sd.query_devices(sd.default.device[1])["name"]
    if sd.default.device[1] is not None
    else "default"
)

print(f"\nInput Device={current_in}")


###setari generale

# ~~~~~CORECTIE MICROFON: intrebare la inceput de tot~~~~~
# Raspunsul se citeste O SINGURA DATA, aici, inainte de orice alta configurare.
# APLICA_CORECTIE_MICROFON e folosit mai jos, la sectiunea "CORECTIE CURBA
# MICROFON", ca sa decida daca se incearca deloc incarcarea/construirea
# filtrului de compensare. Daca raspunzi "nu", codul se comporta identic cu
# inainte de a exista aceasta functionalitate (sos_corectie_mic ramane None).

print("~~~~~CORECTIE MICROFON~~~~~")
raspuns_corectie_mic = input(
    "Aplici corectia de raspuns in frecventa a microfonului (din fisierul de calibrare)? (d/n) > "
).strip().lower()
APLICA_CORECTIE_MICROFON = raspuns_corectie_mic in ("d", "da", "y", "yes")
if APLICA_CORECTIE_MICROFON:
    print("Corectie de microfon: ACTIVATA (se incarca mai jos, dupa ce SAMPLE_RATE e stabilit).")
else:
    print("Corectie de microfon: dezactivata.")

SURSA_OPT=2

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
