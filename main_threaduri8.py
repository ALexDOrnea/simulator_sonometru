# ==========================================================
# SONOMETRU OPTIMIZAT REALTIME
# Modificari:
# - blocksize fix 512
# - latency high stabil
# - queue limitata pentru evitarea acumularii de delay
# - procesare audio float32
# - GUI 30 FPS
# ==========================================================

import warnings
import os
import sys
import time
import queue
import threading
from collections import deque
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfilt_zi, butter, bilinear_zpk, zpk2sos, lfilter

## pentru ignorare avertismente wav
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.io.wavfile")

###########################################
############Selectie in/out################

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Sounddevice nu a fost initializat. incearca sa il instalezi")
    sys.exit()

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
except ImportError:
    print("pyqtgraph nu a fost gasit. instaleaza cu: pip install pyqtgraph pyqt5")
    sys.exit()

input_device_id = None
output_device_id = None

if HAS_SOUNDDEVICE:
    print("\n\n~~~~~CONFIG AUDIO~~~~~")
    devices = sd.query_devices()

    print("\nOptiuni input")
    input_indices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']} (Input channels: {dev['max_input_channels']})")
            input_indices.append(i)

    try:
        id_in = input("select device(blank for default)\n> ").strip()
        if id_in:
            id_in = int(id_in)
            if id_in in input_indices:
                input_device_id = id_in
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    print("\nOptiuni output")
    output_indices = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            print(f"[{i}] {dev['name']} (Output channels: {dev['max_output_channels']})")
            output_indices.append(i)

    try:
        id_out = input("select device(blank for default)\n> ").strip()
        if id_out:
            id_out = int(id_out)
            if id_out in output_indices:
                output_device_id = id_out
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    sd.default.device = (input_device_id, output_device_id)
    current_in = (
        sd.query_devices(sd.default.device[0])["name"]
        if sd.default.device[0] is not None
        else "default"
    )
    current_out = (
        sd.query_devices(sd.default.device[1])["name"]
        if sd.default.device[1] is not None
        else "default"
    )
    print(f"\n\nConfiguration:\nInput={current_in}\nOutput={current_out}\n")

###########################################
############Alte setari####################

print("~~~~~CONFIGURARE SURSA SEMNAL~~~~~")
print("1. WAV")
print("2. Live")
SURSA_OPT = input("> ").strip()

WAV_PATH = None
if SURSA_OPT == "1":
    print("~~~~~CONFIGURARE PATH WAV~~~~~")
    WAV_PATH = input("path to wav file: ").strip().strip('"')
elif SURSA_OPT != "2":
    print("Invalid option")
    sys.exit()

print("~~~~~CONFIGURARE MOD~~~~~")
MODE = input("Mode(Fast, Slow, Peak)\n>  ").strip()

PEAK_MODE = MODE.strip().lower() == "peak"

print("\n~~~~~SELECTIE PONDERARE (WEIGHTING)~~~~~")
print("1 A-Weighting (IEC 61672)")
print("2 C-Weighting (IEC 61672)")
print("3 Z-Weighting (flat, semnal nemodificat)")
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
    print("Optiune invalida. se foloseste A-Weighting")
    TIP_PONDERARE = "A-Weighting"
    SIMBOL_PONDERARE = "A"

print("\n~~~~~SELECTIE ANALIZA PE BENZI (FILTRU DE BANDA)~~~~~")
print("1 Octave intregi   (1/1 octava)")
print("2 Terte de octava  (1/3 octava)")
print("3 Sesimi de octava (1/6 octava)")
print("4 Doisprezecimi    (1/12 octava)")
BANDA_OPT = input("> ").strip()

FRACTIE_MAP = {"1": 1, "2": 3, "3": 6, "4": 12}
if BANDA_OPT not in FRACTIE_MAP:
    print("Optiune invalida. se foloseste 1/3 octava")
FRACTIE_OCTAVA = FRACTIE_MAP.get(BANDA_OPT, 3)

if PEAK_MODE:
    GRAFIC_OPT = None
    print("\n~~~~~MOD PEAK~~~~~")
    print("Se va afisa un live meter (fara grafice) cu maximul in dB, nefiltrat (Z) si ponderat.")
else:
    print("\n~~~~~SELECTIE GRAFICE~~~~~")
    print("1 dB FS (nivel in timp)")
    print("2 FFT (spectru continuu)")
    print("3 Benzi de octava (bare)")
    print("4 Toate")
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

# ~~~~~CONFIGURARE BLOCKSIZE / LATENCY (sd.OutputStream / sd.InputStream)~~~~~
print("\n~~~~~CONFIG AUDIO OPTIMIZAT~~~~~")
# Setari fixe pentru stabilitate realtime DSP
# NOTA: LATENCY="high" cerea driver-ului un buffer intern mare, ceea ce
# adauga intarziere suplimentara peste blocksize. Pentru delay minim, se
# foloseste "low" - acum e sigur, pentru ca threadul de procesare a fost
# optimizat sa nu mai ramana in urma (vezi BATCH_FACTOR mai jos).
# BLOCKSIZE a fost coborat de la 512 la 256 esantioane (~5.3ms in loc de
# ~10.7ms la 48kHz) - latenta de baza a stream-ului scade la jumatate.
# Threadul audio face un singur filtru sos ieftin per bloc, deci ramane usor.
# Daca pe sistemul tau apar xrun-uri/pocnete, urca inapoi la 512 (sau 384).
BLOCKSIZE = 256
LATENCY = "low"
print("Blocksize fix: 256 samples")
print("Latency fix: low")
print(f"Blocksize folosit: {BLOCKSIZE if BLOCKSIZE else 'auto'} | Latency folosita: {LATENCY}")

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE = 48000
EPSILON = 1e-12
AUDIO_NORM = None

# ~~~~~ARHITECTURA PE 3 THREADURI~~~~~
# 1) Thread AUDIO (rulat de PortAudio/sounddevice): doar filtrarea de ponderare
#    (necesara pentru semnalul de iesire la playback) + trimitere in raw_queue.
#    Trebuie sa ramana cat mai usor ca sa nu produca xrun-uri.
# 2) Thread PROCESARE (creat manual, processing_loop): scoate din raw_queue,
#    calculeaza Leq, nivelul ponderat in timp, benzile de octava, FFT-ul de
#    afisare, peak-ul - tot ce e costisitor mutat aici, in afara timpului real.
# 3) Thread GUI (main thread, Qt event loop): doar deseneaza ce vine prin data_queue.
raw_queue = queue.Queue(maxsize=4)      # audio thread -> processing thread
data_queue = queue.Queue()     # processing thread -> GUI thread
stop_event = threading.Event()

# ~~~~~GUI_REFRESH_MS: intervalul real al timerului de desenare~~~~~
# Coborat de la 33ms (~30fps) la 16ms (~60fps): rezultatele calculate stau
# mai putin timp "in asteptare" pana ajung pe ecran. BATCH_FACTOR de mai jos
# se leaga de aceeasi valoare, ca sa ramana sincronizate.
GUI_REFRESH_MS = 16

# ~~~~~BATCH_FACTOR: rata de recalcul a benzilor de octava + FFT~~~~~
# NOTA (dupa profiling): banca de filtre pe benzi de octava (proceseaza_benzi)
# ramanea, si dupa batching-ul legat de GUI_REFRESH_MS (60fps), ~81% din tot
# timpul threadului de procesare. Motivul: fiecare banda trece deja printr-un
# filtru cu constanta de timp Fast=125ms/Slow=1s (TAU_TIMP) - valoarea abia
# se misca intre doua calcule facute la 60Hz, deci recalculul la rata GUI-ului
# arunca majoritatea rezultatelor nefolosite (un RTA hardware tipic actualizeaza
# la 10-20Hz, nu la 60). BENZI_FFT_REFRESH_HZ decupleaza deci rata de calcul a
# benzilor/FFT de la timer-ul GUI (care ramane rapid, la GUI_REFRESH_MS, doar
# pentru curba dB - ieftina, O(1) per punct) si o leaga de o rata proprie,
# suficienta pentru ochi. Rezultatul e matematic IDENTIC (filtre LTI, zi trece
# corect intre blocuri), doar frecventa de recalcul scade semnificativ.
BENZI_FFT_REFRESH_HZ = 20.0
BATCH_FACTOR = max(1, round((1.0 / BENZI_FFT_REFRESH_HZ) * SAMPLE_RATE / BLOCKSIZE))

play_pointer = 0
NIVEL_PODEA = -120.0 + CALIBRARE_DB
NIVEL_PLAFON = 20.0 + CALIBRARE_DB

###########################################
#########CITIRE WAV SI PREGATIRE SEMNAL####

if SURSA_OPT == "1":
    if not os.path.exists(WAV_PATH):
        print(f"{WAV_PATH} not found")
        sys.exit()

    SAMPLE_RATE, AUDIO_DATA = wavfile.read(WAV_PATH)
    if AUDIO_DATA.ndim > 1:
        AUDIO_DATA = AUDIO_DATA[:, 0]
        print("Audio transformed to mono")
    if AUDIO_DATA.dtype == np.int16:
        AUDIO_NORM = AUDIO_DATA.astype(np.float64) / 32768.0
    elif AUDIO_DATA.dtype == np.int32:
        AUDIO_NORM = AUDIO_DATA.astype(np.float64) / 2147483648.0
    elif np.issubdtype(AUDIO_DATA.dtype, np.integer):
        info = np.iinfo(AUDIO_DATA.dtype)
        AUDIO_NORM = AUDIO_DATA.astype(np.float64) / max(abs(info.min), info.max)
    else:
        AUDIO_NORM = AUDIO_DATA.astype(np.float64)

nyquist = SAMPLE_RATE / 2.0

########################################################
############# FILTRE DE PONDERARE (IEC 61672) ##########
########################################################

def get_a_weighting_filter(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    p1, p2, p3, p4 = -2*np.pi*f1, -2*np.pi*f2, -2*np.pi*f3, -2*np.pi*f4
    z = [0, 0, 0, 0]
    p = [p1, p1, p2, p3, p4, p4]
    k = (2 * np.pi * f4)**2 * (10**(A1000 / 20))
    zeros_d, poles_d, gain_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(zeros_d, poles_d, gain_d)

def get_c_weighting_filter(fs):
    f1, f4 = 20.598997, 12194.217
    C1000 = 0.0619
    p1, p4 = -2*np.pi*f1, -2*np.pi*f4
    z = [0, 0]
    p = [p1, p1, p4, p4]
    k = (2 * np.pi * f4)**2 * (10**(C1000 / 20))
    zeros_d, poles_d, gain_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(zeros_d, poles_d, gain_d)

if TIP_PONDERARE == "A-Weighting":
    sos_ponderare = get_a_weighting_filter(SAMPLE_RATE)
    zi_ponderare = sosfilt_zi(sos_ponderare) * 0.0
elif TIP_PONDERARE == "C-Weighting":
    sos_ponderare = get_c_weighting_filter(SAMPLE_RATE)
    zi_ponderare = sosfilt_zi(sos_ponderare) * 0.0
else:  # Z-Weighting: fara filtrare, raspuns flat
    sos_ponderare = None
    zi_ponderare = None

def filtreaza_block(chunk):
    """Aplica ponderarea de frecventa selectata (A, C sau Z) semnalului brut.
    Ramane in threadul AUDIO pentru ca la playback rezultatul e chiar semnalul
    care se aude - trebuie calculat 'live', in callback. E un singur filtru sos,
    deci e ieftin si nu pune presiune pe bugetul de timp real."""
    global zi_ponderare
    if sos_ponderare is None:
        return chunk.copy()
    chunk_ponderat, zi_ponderare = sosfilt(sos_ponderare, chunk, zi=zi_ponderare)
    return chunk_ponderat

########################################################
### PONDERARE EXPONENTIALA IN TIMP (IEC 61672-1, 3.6) ###
########################################################

def creeaza_filtru_timp(tau, fs):
    alpha = 1.0 - np.exp(-1.0 / (tau * fs))
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    return b, a

if not PEAK_MODE:
    TAU_TIMP = 1.0 if MODE.strip().lower() == "slow" else 0.125
    if MODE.strip().lower() not in ("fast", "slow"):
        print("not a valid time-weighting mode. using fast")
        TAU_TIMP = 0.125
else:
    TAU_TIMP = 0.125

b_timp, a_timp = creeaza_filtru_timp(TAU_TIMP, SAMPLE_RATE)
zi_nivel_raw = [0.0]        # NOTA: acum atins DOAR din threadul de PROCESARE
zi_nivel_filt = [0.0]       # NOTA: acum atins DOAR din threadul de PROCESARE

def nivel_ponderat_in_timp(chunk, b, a, zi):
    ms, zi_nou = lfilter(b, a, np.square(chunk), zi=zi)
    return ms, zi_nou

def db_din_ms(valoare_ms):
    return float(np.clip(10.0 * np.log10(valoare_ms + EPSILON) + CALIBRARE_DB, NIVEL_PODEA, NIVEL_PLAFON))

########################################################
############ Leq (time-averaged sound level) ###########
########################################################

leq_state = {
    "suma_patrate_raw": 0.0,
    "suma_patrate_filt": 0.0,
    "n_esantioane": 0,
}
leq_lock = threading.Lock()  # protejeaza resetarea din threadul GUI (buton) vs. actualizarea din threadul de procesare

def actualizeaza_leq(chunk_raw, chunk_ponderat):
    with leq_lock:
        leq_state["suma_patrate_raw"] += float(np.sum(np.square(chunk_raw)))
        leq_state["suma_patrate_filt"] += float(np.sum(np.square(chunk_ponderat)))
        leq_state["n_esantioane"] += len(chunk_raw)

def calculeaza_leq():
    with leq_lock:
        n = leq_state["n_esantioane"]
        if n == 0:
            return NIVEL_PODEA, NIVEL_PODEA
        mp_raw = leq_state["suma_patrate_raw"] / n
        mp_filt = leq_state["suma_patrate_filt"] / n
    l_zeq = db_din_ms(mp_raw)
    l_xeq = db_din_ms(mp_filt)
    return l_zeq, l_xeq

def reseteaza_leq():
    with leq_lock:
        leq_state["suma_patrate_raw"] = 0.0
        leq_state["suma_patrate_filt"] = 0.0
        leq_state["n_esantioane"] = 0

########################################################
##### BANC DE FILTRE PE BENZI DE OCTAVA (IEC 61260-1) ###
########################################################

OCTAVE_G = 10 ** (3.0 / 10.0)
FREQ_REF = 1000.0

def formateaza_frecventa(f):
    if f >= 1000:
        return f"{f/1000:.3g}k"
    return f"{f:.3g}"

def frecventa_centrala_exacta(x, b):
    if b % 2 == 1:
        return FREQ_REF * OCTAVE_G ** (x / b)
    else:
        return FREQ_REF * OCTAVE_G ** ((2 * x + 1) / (2 * b))

def design_octave_bands(fs, b, fmin=20.0, fmax=20000.0):
    nyq = fs / 2.0
    fmax = min(fmax, nyq * 0.98)
    ordin = 12

    x_min = int(np.floor(b * np.log(fmin / FREQ_REF) / np.log(OCTAVE_G))) - 1
    x_max = int(np.ceil(b * np.log(fmax / FREQ_REF) / np.log(OCTAVE_G))) + 1

    benzi = []
    for x in range(x_min, x_max + 1):
        fc = frecventa_centrala_exacta(x, b)
        if fc < fmin * 0.95 or fc > fmax * 1.05:
            continue
        f_low = fc * OCTAVE_G ** (-1.0 / (2 * b))
        f_high = fc * OCTAVE_G ** (1.0 / (2 * b))
        if f_low < 1.0 or f_high >= nyq * 0.98:
            continue
        sos = butter(ordin, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        zi_bandpass = sosfilt_zi(sos) * 0.0
        b_timp_banda, a_timp_banda = creeaza_filtru_timp(TAU_TIMP, fs)
        zi_timp = [0.0]
        benzi.append({
            "x": x, "fc": fc, "f_low": f_low, "f_high": f_high,
            "sos": sos, "zi_bandpass": zi_bandpass,
            "b_timp": b_timp_banda, "a_timp": a_timp_banda, "zi_timp": zi_timp,
        })
    return benzi

octave_bands = design_octave_bands(SAMPLE_RATE, FRACTIE_OCTAVA)
NUM_BENZI = len(octave_bands)

if NUM_BENZI == 0:
    print("Nu s-a putut genera nicio banda de frecventa pentru configuratia curenta.")
    sys.exit()

print(
    f"\nS-au generat {NUM_BENZI} benzi de 1/{FRACTIE_OCTAVA} octava "
    f"intre {octave_bands[0]['fc']:.1f} Hz si {octave_bands[-1]['fc']:.0f} Hz "
    f"(ponderare: {TIP_PONDERARE})"
)

def proceseaza_benzi(chunk_ponderat):
    """Cel mai costisitor pas (pana la ~30 filtre ordin 12) - ruleaza EXCLUSIV
    in threadul de procesare, niciodata in callback-ul audio."""
    n = len(chunk_ponderat)
    if n == 0:
        return np.full(NUM_BENZI, NIVEL_PODEA)

    niveluri = np.empty(NUM_BENZI)
    for i, banda in enumerate(octave_bands):
        filtrat, banda["zi_bandpass"] = sosfilt(banda["sos"], chunk_ponderat, zi=banda["zi_bandpass"])
        ms, banda["zi_timp"] = lfilter(banda["b_timp"], banda["a_timp"], np.square(filtrat), zi=banda["zi_timp"])
        niveluri[i] = db_din_ms(ms[-1])
    return niveluri

########################################################
######### FERESTRE PENTRU AFISAREA FFT (DOAR VIZUAL) ###
########################################################

if MODE.lower() == "fast":
    WINDOW_SIZE_FFT = int(0.125 * SAMPLE_RATE)
elif MODE.lower() == "slow":
    WINDOW_SIZE_FFT = int(1.0 * SAMPLE_RATE)
elif MODE.lower() == "peak":
    WINDOW_SIZE_FFT = int(0.035 * SAMPLE_RATE)
else:
    WINDOW_SIZE_FFT = int(0.125 * SAMPLE_RATE)

live_ring_buffer_raw_fft = np.zeros(WINDOW_SIZE_FFT)
live_ring_buffer_filtered_fft = np.zeros(WINDOW_SIZE_FFT)
hanning_window = np.hanning(WINDOW_SIZE_FFT)
fft_frequencies = np.fft.rfftfreq(WINDOW_SIZE_FFT, d=1.0 / SAMPLE_RATE)

FFT_DISPLAY_STEP = max(1, len(fft_frequencies) // 2000)
fft_frequencies_disp = fft_frequencies[::FFT_DISPLAY_STEP]

semnal_nefiltrat_complet = []
semnal_filtrat_complet = []
semnal_lock = threading.Lock()  # cele doua liste sunt scrise din processing_loop si citite la final din main

########################################################
############# PROCESARE (thread dedicat) ###############
########################################################

def actualizeaza_ring_buffer(buffer, chunk):
    frames = len(chunk)
    if frames >= len(buffer):
        buffer[:] = chunk[-len(buffer):]
    else:
        # slicing in loc de np.roll: acelasi rezultat, fara aritmetica modulo
        # si fara sa realoce/rotesca tot bufferul de fiecare data
        buffer[:-frames] = buffer[frames:]
        buffer[-frames:] = chunk

def calculeaza_fft_pentru_afisare(buffer):
    windowed_signal = buffer * hanning_window
    fft_raw = np.abs(np.fft.rfft(windowed_signal))
    fft_norm = fft_raw / (WINDOW_SIZE_FFT / 2.0)
    fft_db = 20 * np.log10(fft_norm + EPSILON) + CALIBRARE_DB
    fft_db = np.clip(fft_db, NIVEL_PODEA, NIVEL_PLAFON)
    return fft_db

def calculeaza_peak_db(buffer):
    peak = np.max(np.abs(buffer))
    val = 20 * np.log10(peak + EPSILON) + CALIBRARE_DB
    return float(np.clip(val, NIVEL_PODEA, NIVEL_PLAFON))

peak_hold_state = {
    "raw_db": NIVEL_PODEA,
    "filt_db": NIVEL_PODEA,
}
peak_lock = threading.Lock()

def get_peak_hold_display(cheie, valoare_curenta):
    val_key = f"{cheie}_db"
    with peak_lock:
        if valoare_curenta > peak_hold_state[val_key]:
            peak_hold_state[val_key] = valoare_curenta
        return peak_hold_state[val_key]

def reseteaza_peak_hold():
    with peak_lock:
        peak_hold_state["raw_db"] = NIVEL_PODEA
        peak_hold_state["filt_db"] = NIVEL_PODEA

_batch_raw = []
_batch_filt = []
_batch_count = 0

def proceseaza_chunk(chunk, chunk_ponderat, pointer_esantion):
    """Tot ce era inainte in trimite_date_live, acum rulat EXCLUSIV in
    threadul de procesare (nu mai atinge deloc threadul audio).

    Nivelul ponderat in timp (bara/curba dB) si Leq raman calculate pe
    FIECARE bloc - sunt ieftine (un singur filtru recursiv de ordin mic) si
    trebuie sa fie fidele in timp. Benzile de octava si FFT-ul de afisare
    (partea scumpa) se calculeaza doar o data la BATCH_FACTOR blocuri, pe
    blocul concatenat - vezi nota de la BATCH_FACTOR mai sus."""
    global zi_nivel_raw, zi_nivel_filt, _batch_count
    current_time = pointer_esantion / SAMPLE_RATE

    actualizeaza_leq(chunk, chunk_ponderat)
    l_zeq, l_xeq = calculeaza_leq()

    if PEAK_MODE:
        db_raw = calculeaza_peak_db(chunk)
        db_filtered = calculeaza_peak_db(chunk_ponderat)
        data_queue.put((current_time, db_raw, db_filtered, None, None, None, l_zeq, l_xeq))
        return

    ms_raw, zi_nivel_raw = nivel_ponderat_in_timp(chunk, b_timp, a_timp, zi_nivel_raw)
    ms_filt, zi_nivel_filt = nivel_ponderat_in_timp(chunk_ponderat, b_timp, a_timp, zi_nivel_filt)
    db_raw = db_din_ms(ms_raw[-1])
    db_filtered = db_din_ms(ms_filt[-1])

    fft_raw = fft_filtered = niveluri_benzi = None

    if plot_fft is not None or plot_bands is not None:
        _batch_raw.append(chunk)
        _batch_filt.append(chunk_ponderat)
        _batch_count += 1

        if _batch_count >= BATCH_FACTOR:
            bloc_raw = np.concatenate(_batch_raw)
            bloc_filt = np.concatenate(_batch_filt)
            _batch_raw.clear()
            _batch_filt.clear()
            _batch_count = 0

            if plot_fft is not None:
                actualizeaza_ring_buffer(live_ring_buffer_raw_fft, bloc_raw)
                actualizeaza_ring_buffer(live_ring_buffer_filtered_fft, bloc_filt)
                fft_raw = calculeaza_fft_pentru_afisare(live_ring_buffer_raw_fft)
                fft_filtered = calculeaza_fft_pentru_afisare(live_ring_buffer_filtered_fft)

            if plot_bands is not None:
                niveluri_benzi = proceseaza_benzi(bloc_filt)

    data_queue.put((
        current_time,
        db_raw,
        db_filtered,
        fft_raw[::FFT_DISPLAY_STEP] if fft_raw is not None else None,
        fft_filtered[::FFT_DISPLAY_STEP] if fft_filtered is not None else None,
        niveluri_benzi,
        l_zeq,
        l_xeq,
    ))

def processing_loop():
    """Bucla threadului de PROCESARE: consuma din raw_queue (umpluta de threadul
    audio) si produce in data_queue (consumata de threadul GUI). Ruleaza pana
    cand stop_event e setat SI coada s-a golit."""
    while True:
        try:
            chunk, chunk_ponderat, pointer_esantion = raw_queue.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        with semnal_lock:
            semnal_nefiltrat_complet.append(chunk)
            semnal_filtrat_complet.append(chunk_ponderat)

        proceseaza_chunk(chunk, chunk_ponderat, pointer_esantion)

########################################################
############# THREAD AUDIO (callback-uri) ###############
########################################################
# ATENTIE: aceste functii ruleaza pe threadul realtime al PortAudio.
# Fac STRICT minimul: filtrarea de ponderare (necesara pentru redare) +
# trimitere in raw_queue. Nimic altceva - nicio operatie costisitoare aici.

def playback_callback(outdata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
    chunk = AUDIO_NORM[play_pointer:play_pointer + frames]
    valid_frames = len(chunk)
    if valid_frames == 0:
        outdata.fill(0)
        raise sd.CallbackStop()
    chunk_ponderat = filtreaza_block(chunk)
    outdata.fill(0)
    outdata[:valid_frames, 0] = chunk_ponderat
    play_pointer += valid_frames
    try:
        raw_queue.put_nowait((chunk.copy(), chunk_ponderat.copy(), play_pointer))
    except queue.Full:
        pass
    if valid_frames < frames:
        raise sd.CallbackStop()

def record_callback(indata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
    chunk = indata[:, 0].astype(np.float32, copy=True)
    chunk_ponderat = filtreaza_block(chunk)
    play_pointer += len(chunk)
    try:
        raw_queue.put_nowait((chunk, chunk_ponderat.copy(), play_pointer))
    except queue.Full:
        pass

########################################################
################ Interfata grafica (pyqtgraph) ##########
########################################################

pg.setConfigOptions(antialias=False, useOpenGL=True, background="k", foreground="w")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

titlu_context_str = f" [{TIP_PONDERARE} | 1/{FRACTIE_OCTAVA} oct | cal {CALIBRARE_DB:+.1f} dB]"

plot_db = None
plot_fft = None
plot_bands = None
curve_db_raw = curve_db_filtered = None
curve_fft_raw = curve_fft_filtered = None
bar_item = None

COL_ORANGE = (255, 165, 0)
COL_PURPLE = (170, 90, 220)
COL_CYAN = (0, 220, 220)
COL_GRAY = (120, 120, 120)
COL_GREEN = (46, 204, 113)

class CurbaIncrementala(pg.GraphicsObject):
    """Curba pentru graficul dB-vs-timp care NU reconstruieste tot path-ul din
    tot arrayul la fiecare frame (asa cum face PlotCurveItem.setData()).
    In loc, tine un QPainterPath persistent si doar ADAUGA punctul nou cu
    path.lineTo() -> cost O(1) per punct nou, nu O(n) din tot istoricul.

    Fereastra glisanta (deque cu maxlen) tot are nevoie, ocazional, de o
    sincronizare cu path-ul ca sa elimine punctele iesite din fereastra -
    dar asta se face rar (reconstruieste_din_date), nu la fiecare frame."""

    def __init__(self, culoare, latime=2):
        super().__init__()
        self.pen = pg.mkPen(culoare, width=latime)
        # LegendItem citeste self.opts['pen'] cand deseneaza liniuta-mostra din legenda -
        # fara acest atribut arunca AttributeError la FIECARE repaint (asta cauza lag-ul uriaș).
        self.opts = {"pen": self.pen}
        self.path = QtGui.QPainterPath()
        self._are_punct_initial = False
        self._bounding_rect = QtCore.QRectF()

    def adauga_punct(self, x, y):
        if not self._are_punct_initial:
            self.path.moveTo(x, y)
            self._are_punct_initial = True
        else:
            self.path.lineTo(x, y)
        self._bounding_rect = self._bounding_rect.united(QtCore.QRectF(x, y, 0.0001, 0.0001))
        self.prepareGeometryChange()
        self.update()

    def reconstruieste_din_date(self, x_iterabil, y_iterabil):
        """Reface path-ul complet, o singura data - folosit periodic (nu la
        fiecare frame) ca sa sincronizeze curba cu fereastra glisanta din deque
        (elimina punctele expirate din stanga)."""
        path_nou = QtGui.QPainterPath()
        prim = True
        rect = QtCore.QRectF()
        for x, y in zip(x_iterabil, y_iterabil):
            if prim:
                path_nou.moveTo(x, y)
                prim = False
            else:
                path_nou.lineTo(x, y)
            rect = rect.united(QtCore.QRectF(x, y, 0.0001, 0.0001))
        self.path = path_nou
        self._are_punct_initial = not prim
        self._bounding_rect = rect
        self.prepareGeometryChange()
        self.update()

    def reseteaza(self):
        self.path = QtGui.QPainterPath()
        self._are_punct_initial = False
        self._bounding_rect = QtCore.QRectF()
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen)
        painter.drawPath(self.path)

def culoare_pentru_nivel(db):
    prag_rosu = -6.0 + CALIBRARE_DB
    prag_galben = -18.0 + CALIBRARE_DB
    if db > prag_rosu:
        return "#e74c3c"
    elif db > prag_galben:
        return "#f1c40f"
    else:
        return "#2ecc71"

def creeaza_coloana_meter(nume):
    coloana = QtWidgets.QVBoxLayout()

    eticheta_nume = QtWidgets.QLabel(nume)
    eticheta_nume.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
    eticheta_nume.setAlignment(QtCore.Qt.AlignCenter)

    bara = QtWidgets.QProgressBar()
    bara.setOrientation(QtCore.Qt.Vertical)
    bara.setRange(0, 1200)
    bara.setValue(0)
    bara.setTextVisible(False)
    bara.setFixedWidth(50)
    bara.setStyleSheet(
        "QProgressBar { background-color: #222; border: 1px solid #555; }"
        "QProgressBar::chunk { background-color: #2ecc71; }"
    )

    eticheta_valoare = QtWidgets.QLabel(f"{NIVEL_PODEA:.1f} dB")
    eticheta_valoare.setStyleSheet("color: white; font-size: 12px;")
    eticheta_valoare.setAlignment(QtCore.Qt.AlignCenter)

    eticheta_peak = QtWidgets.QLabel(f"Peak: {NIVEL_PODEA:.1f} dB")
    eticheta_peak.setStyleSheet("color: #f1c40f; font-size: 11px;")
    eticheta_peak.setAlignment(QtCore.Qt.AlignCenter)

    eticheta_leq = QtWidgets.QLabel(f"Leq: {NIVEL_PODEA:.1f} dB")
    eticheta_leq.setStyleSheet("color: #3498db; font-size: 11px;")
    eticheta_leq.setAlignment(QtCore.Qt.AlignCenter)

    coloana.addWidget(eticheta_nume)
    coloana.addWidget(bara, alignment=QtCore.Qt.AlignHCenter)
    coloana.addWidget(eticheta_valoare)
    coloana.addWidget(eticheta_peak)
    coloana.addWidget(eticheta_leq)
    return coloana, bara, eticheta_valoare, eticheta_peak, eticheta_leq

if PEAK_MODE:
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Live Peak Meter{titlu_context_str}")
    win.resize(360, 620)
    win.setStyleSheet("background-color: black;")

    layout_vertical = QtWidgets.QVBoxLayout(win)
    layout_principal = QtWidgets.QHBoxLayout()

    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw, eticheta_leq_raw = creeaza_coloana_meter("Z (nefiltrat)")
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt, eticheta_leq_filt = creeaza_coloana_meter(
        f"{TIP_PONDERARE}"
    )

    layout_principal.addLayout(coloana_raw)
    layout_principal.addLayout(coloana_filt)

    layout_butoane = QtWidgets.QHBoxLayout()
    buton_reset_peak = QtWidgets.QPushButton("Reset Peak")
    buton_reset_peak.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 8px; font-size: 13px; }"
        "QPushButton:hover { background-color: #555; }"
        "QPushButton:pressed { background-color: #777; }"
    )
    buton_reset_peak.clicked.connect(reseteaza_peak_hold)

    buton_reset_leq = QtWidgets.QPushButton("Reset Leq")
    buton_reset_leq.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 8px; font-size: 13px; }"
        "QPushButton:hover { background-color: #555; }"
        "QPushButton:pressed { background-color: #777; }"
    )
    buton_reset_leq.clicked.connect(reseteaza_leq)

    layout_butoane.addWidget(buton_reset_peak)
    layout_butoane.addWidget(buton_reset_leq)

    layout_vertical.addLayout(layout_principal)
    layout_vertical.addLayout(layout_butoane)

    win.show()

else:
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Analiza DSP live{titlu_context_str} - Mod: {MODE}")
    win.resize(1280, 960)
    win.setStyleSheet("background-color: black;")

    layout_main = QtWidgets.QHBoxLayout(win)

    win_graphics = pg.GraphicsLayoutWidget()
    layout_main.addWidget(win_graphics, stretch=4)

    panel_meter = QtWidgets.QWidget()
    layout_meter = QtWidgets.QVBoxLayout(panel_meter)

    layout_bars = QtWidgets.QHBoxLayout()
    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw, eticheta_leq_raw = creeaza_coloana_meter("Z (nefiltrat)")
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt, eticheta_leq_filt = creeaza_coloana_meter(
        f"{TIP_PONDERARE}"
    )

    layout_bars.addLayout(coloana_raw)
    layout_bars.addLayout(coloana_filt)

    buton_reset_peak = QtWidgets.QPushButton("Reset Peak")
    buton_reset_peak.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 6px; font-size: 11px; }"
        "QPushButton:hover { background-color: #555; }"
    )
    buton_reset_peak.clicked.connect(reseteaza_peak_hold)

    buton_reset_leq = QtWidgets.QPushButton("Reset Leq")
    buton_reset_leq.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 6px; font-size: 11px; }"
        "QPushButton:hover { background-color: #555; }"
    )
    buton_reset_leq.clicked.connect(reseteaza_leq)

    layout_meter.addLayout(layout_bars)
    layout_meter.addWidget(buton_reset_peak)
    layout_meter.addWidget(buton_reset_leq)
    layout_main.addWidget(panel_meter, stretch=1)

    win.show()

    afiseaza_db = GRAFIC_OPT in ("1", "4")
    afiseaza_fft = GRAFIC_OPT in ("2", "4")
    afiseaza_benzi = GRAFIC_OPT in ("3", "4")

    if afiseaza_db:
        plot_db = win_graphics.addPlot(title=f"Nivel live: Z (nefiltrat) vs {TIP_PONDERARE}")
        plot_db.setLabel("bottom", "Timp (s)")
        plot_db.setLabel("left", "Nivel (dB SPL)")
        plot_db.setYRange(NIVEL_PODEA, NIVEL_PLAFON)
        plot_db.showGrid(x=True, y=True, alpha=0.3)
        plot_db.addLegend()
        curve_db_raw = CurbaIncrementala(COL_ORANGE, latime=2)
        curve_db_filtered = CurbaIncrementala(COL_PURPLE, latime=2)
        plot_db.addItem(curve_db_raw)
        plot_db.addItem(curve_db_filtered)
        plot_db.legend.addItem(curve_db_raw, f"Z (nefiltrat) - nivel ({MODE}, IEC 61672-1 Ec.1)")
        plot_db.legend.addItem(curve_db_filtered, f"{TIP_PONDERARE} - nivel ({MODE}, IEC 61672-1 Ec.1)")

        if SURSA_OPT == "1":
            duration = len(AUDIO_NORM) / SAMPLE_RATE
            plot_db.setXRange(0, duration)
        else:
            plot_db.setXRange(0, 10)

        if afiseaza_fft or afiseaza_benzi:
            win_graphics.nextRow()

    if afiseaza_fft:
        plot_fft = win_graphics.addPlot(title=f"FFT in timp real: Z (nefiltrat) vs {TIP_PONDERARE}")
        plot_fft.setLabel("bottom", "Frecventa (Hz)")
        plot_fft.setLabel("left", "Amplitudine (dB SPL)")
        plot_fft.setLogMode(x=True, y=False)
        plot_fft.setYRange(NIVEL_PODEA, NIVEL_PLAFON)
        plot_fft.showGrid(x=True, y=True, alpha=0.3)
        plot_fft.addLegend()

        max_plot_frequency = min(20000, nyquist)
        min_plot_frequency = min(20, max_plot_frequency / 10)
        plot_fft.setXRange(np.log10(min_plot_frequency), np.log10(max_plot_frequency))

        curve_fft_raw = plot_fft.plot(pen=pg.mkPen(COL_ORANGE, width=1.5), name="FFT Z (nefiltrat)")
        curve_fft_filtered = plot_fft.plot(pen=pg.mkPen(COL_CYAN, width=1.5), name=f"FFT {TIP_PONDERARE}")

        if afiseaza_benzi:
            win_graphics.nextRow()

    if afiseaza_benzi:
        plot_bands = win_graphics.addPlot(
            title=f"Analiza pe benzi de 1/{FRACTIE_OCTAVA} octava ({TIP_PONDERARE}) - IEC 61260-1"
        )
        plot_bands.setLabel("bottom", "Frecventa centrala banda (Hz)")
        plot_bands.setLabel("left", "Nivel (dB SPL)")
        plot_bands.setYRange(NIVEL_PODEA, NIVEL_PLAFON)
        plot_bands.showGrid(x=False, y=True, alpha=0.3)

        x_positions = np.arange(NUM_BENZI)
        bar_item = pg.BarGraphItem(
            x=x_positions, height=np.zeros(NUM_BENZI), width=0.8, brush=COL_GREEN,
            y0=NIVEL_PODEA
        )
        plot_bands.addItem(bar_item)

        ticks = []
        for i, banda in enumerate(octave_bands):
            if banda["x"] % FRACTIE_OCTAVA == 0:
                ticks.append((i, formateaza_frecventa(banda["fc"])))
        plot_bands.getAxis("bottom").setTicks([ticks])
        plot_bands.setXRange(-1, NUM_BENZI)

# In modul Live, doar ultimele 10s se vad oricum in fereastra (setXRange mai jos) -
# un deque cu maxlen elimina cresterea nemarginita a listelor si costul tot mai
# mare al setData() pe masura ce trece timpul. In modul WAV, istoricul e oricum
# marginit de durata fisierului, deci pastram lista completa.
if SURSA_OPT == "2" and not PEAK_MODE:
    _puncte_pe_secunda = max(1, SAMPLE_RATE // BLOCKSIZE) if BLOCKSIZE else 200
    _MAXLEN_LIVE = int(_puncte_pe_secunda * 15)  # ~15s de istoric, cu marja fata de fereastra de 10s afisata
    x_data = deque(maxlen=_MAXLEN_LIVE)
    y_db_raw = deque(maxlen=_MAXLEN_LIVE)
    y_db_filtered = deque(maxlen=_MAXLEN_LIVE)
    # reconstruim path-ul curbei incrementale din deque o data la atatea puncte noi -
    # nu la fiecare frame - ca sa "taiem" din stanga punctele expirate din fereastra.
    REBUILD_LA_PUNCTE = max(30, _MAXLEN_LIVE // 15)
else:
    x_data = []
    y_db_raw = []
    y_db_filtered = []
    REBUILD_LA_PUNCTE = None

_puncte_de_la_ultima_reconstructie = 0

########################################################
#################PORNIREA STREAMULUI####################

stream = None
running = {"active": True}

def on_close(event=None):
    running["active"] = False

win.closeEvent = lambda event: (on_close(), event.accept())

def actualizeaza_bara(bara, eticheta_valoare, eticheta_peak, eticheta_leq_lbl, valoare_db, valoare_peak_db, valoare_leq_db):
    interval_bara = NIVEL_PLAFON - NIVEL_PODEA
    poz = int(np.clip((valoare_db - NIVEL_PODEA) / interval_bara * 1200, 0, 1200))
    bara.setValue(poz)
    bara.setStyleSheet(
        "QProgressBar { background-color: #222; border: 1px solid #555; }"
        f"QProgressBar::chunk {{ background-color: {culoare_pentru_nivel(valoare_db)}; }}"
    )
    eticheta_valoare.setText(f"{valoare_db:6.1f} dB")
    eticheta_peak.setText(f"Peak: {valoare_peak_db:6.1f} dB")
    eticheta_leq_lbl.setText(f"Leq: {valoare_leq_db:6.1f} dB")

def update_gui():
    """Threadul GUI: NU mai face nicio filtrare/procesare, doar citeste
    din data_queue (umpluta de threadul de procesare) si deseneaza."""
    global _puncte_de_la_ultima_reconstructie
    if not running["active"]:
        return

    updated = False
    last_time = None
    last_db_raw = None
    last_db_filtered = None
    last_fft_raw = None
    last_fft_filtered = None
    last_niveluri_benzi = None
    last_leq_raw = None
    last_leq_filt = None

    while True:
        try:
            (t, db_raw, db_filtered, fft_raw, fft_filtered, niveluri_benzi, leq_raw, leq_filt) = data_queue.get_nowait()
            last_time = t
            last_db_raw = db_raw
            last_db_filtered = db_filtered
            if not PEAK_MODE:
                x_data.append(t)
                y_db_raw.append(db_raw)
                y_db_filtered.append(db_filtered)
                if plot_db is not None:
                    # ADAUGARE INCREMENTALA: doar punctul nou, nu tot arrayul (vezi CurbaIncrementala)
                    curve_db_raw.adauga_punct(t, db_raw)
                    curve_db_filtered.adauga_punct(t, db_filtered)
                    if REBUILD_LA_PUNCTE is not None:
                        _puncte_de_la_ultima_reconstructie += 1
                        if _puncte_de_la_ultima_reconstructie >= REBUILD_LA_PUNCTE:
                            # sincronizare rara (nu la fiecare frame) cu fereastra glisanta din deque -
                            # aici "taiem" punctele care au iesit din fereastra
                            curve_db_raw.reconstruieste_din_date(x_data, y_db_raw)
                            curve_db_filtered.reconstruieste_din_date(x_data, y_db_filtered)
                            _puncte_de_la_ultima_reconstructie = 0
            # fft/niveluri_benzi vin doar o data la BATCH_FACTOR elemente din coada
            # (vezi proceseaza_chunk) - nu le suprascriem cu None cand un element
            # ulterior din aceeasi golire a cozii nu are date noi de benzi/FFT.
            if fft_raw is not None:
                last_fft_raw = fft_raw
                last_fft_filtered = fft_filtered
            if niveluri_benzi is not None:
                last_niveluri_benzi = niveluri_benzi
            last_leq_raw = leq_raw
            last_leq_filt = leq_filt
            updated = True
        except queue.Empty:
            break

    if not updated:
        if stream is not None and not stream.active and raw_queue.empty() and data_queue.empty():
            running["active"] = False
            app.quit()
        return

    peak_raw = get_peak_hold_display("raw", last_db_raw)
    peak_filt = get_peak_hold_display("filt", last_db_filtered)
    actualizeaza_bara(bara_raw, eticheta_raw, eticheta_peak_raw, eticheta_leq_raw, last_db_raw, peak_raw, last_leq_raw)
    actualizeaza_bara(bara_filt, eticheta_filt, eticheta_peak_filt, eticheta_leq_filt, last_db_filtered, peak_filt, last_leq_filt)

    if PEAK_MODE:
        print(
            f"Timp: {last_time:6.2f}s | "
            f"Z (nefiltrat): {last_db_raw:6.1f} dB SPL (peak {peak_raw:6.1f}, Leq {last_leq_raw:6.1f}) | "
            f"{TIP_PONDERARE}: {last_db_filtered:6.1f} dB SPL (peak {peak_filt:6.1f}, Leq {last_leq_filt:6.1f})"
        )
    else:
        print(f"Timp: {x_data[-1]:6.2f}s | Z (nefiltrat): {y_db_raw[-1]:6.1f} dB SPL (Leq {last_leq_raw:6.1f}) | {TIP_PONDERARE}: {y_db_filtered[-1]:6.1f} dB SPL (Leq {last_leq_filt:6.1f})")

        if plot_db is not None:
            if SURSA_OPT == "2" and x_data[-1] > 10:
                plot_db.setXRange(x_data[-1] - 10, x_data[-1], padding=0)
            # NOTA: curve_db_raw/curve_db_filtered au fost deja actualizate incremental
            # mai sus (adauga_punct), nu mai e nevoie de un setData() cu tot arrayul aici.

        if plot_fft is not None and last_fft_raw is not None:
            curve_fft_raw.setData(fft_frequencies_disp, last_fft_raw)
            curve_fft_filtered.setData(fft_frequencies_disp, last_fft_filtered)

        if plot_bands is not None and last_niveluri_benzi is not None:
            bar_item.setOpts(height=last_niveluri_benzi - NIVEL_PODEA)

    if stream is not None and not stream.active and raw_queue.empty() and data_queue.empty():
        running["active"] = False
        app.quit()

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(GUI_REFRESH_MS)  # ~60 fps

processing_thread = threading.Thread(target=processing_loop, name="ProcesareDSP", daemon=True)
processing_thread.start()

try:
    if SURSA_OPT == "1":
        print(f"se reda audio ponderat {TIP_PONDERARE} si se afiseaza comparatia live + benzi de 1/{FRACTIE_OCTAVA} octava")
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            latency=LATENCY,
            callback=playback_callback,
        )
    else:
        print(f"se analizeaza microfonul cu ponderare {TIP_PONDERARE} si se afiseaza comparatia live + benzi de 1/{FRACTIE_OCTAVA} octava")
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            latency=LATENCY,
            callback=record_callback,
        )

    with stream:
        app.exec_()

except KeyboardInterrupt:
    print("\nmonitorizare oprita de utilizator.")

finally:
    if HAS_SOUNDDEVICE:
        sd.stop()

    # oprim threadul de procesare abia dupa ce stream-ul s-a inchis, ca sa
    # apuce sa consume tot ce a mai ramas in raw_queue (ultimele chunk-uri)
    stop_event.set()
    processing_thread.join(timeout=2.0)

    print("\nprocesare finalizata")

    l_zeq_final, l_xeq_final = calculeaza_leq()
    print(f"Leq final (durata masurata): L_Zeq = {l_zeq_final:.1f} dB | L_{SIMBOL_PONDERARE}eq = {l_xeq_final:.1f} dB")

    with semnal_lock:
        avem_semnal = bool(semnal_nefiltrat_complet) and not PEAK_MODE
        if avem_semnal:
            semnal_complet_raw = np.concatenate(semnal_nefiltrat_complet)
            semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

    if avem_semnal:
        N = len(semnal_complet_raw)
        freqs_finale = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)

        final_window = np.hanning(N)
        fft_orig = np.abs(np.fft.rfft(semnal_complet_raw * final_window)) / (N / 2.0)
        fft_filt = np.abs(np.fft.rfft(semnal_complet_filt * final_window)) / (N / 2.0)

        db_orig = np.clip(
            20 * np.log10(fft_orig + EPSILON) + CALIBRARE_DB,
            NIVEL_PODEA, NIVEL_PLAFON
        )
        db_filt = np.clip(
            20 * np.log10(fft_filt + EPSILON) + CALIBRARE_DB,
            NIVEL_PODEA, NIVEL_PLAFON
        )

        FINAL_DISPLAY_STEP = max(1, len(freqs_finale) // 4000)
        freqs_disp = freqs_finale[::FINAL_DISPLAY_STEP]
        db_orig_disp = db_orig[::FINAL_DISPLAY_STEP]
        db_filt_disp = db_filt[::FINAL_DISPLAY_STEP]

        titlu_final = f"comparatie spectrala: Z (nefiltrat) vs {TIP_PONDERARE} (cal {CALIBRARE_DB:+.1f} dB)"

        app2 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        win2 = pg.GraphicsLayoutWidget(title=titlu_final)
        win2.resize(1100, 500)
        win2.show()

        plot_comp = win2.addPlot(title=titlu_final)
        plot_comp.setLabel("bottom", "Frecventa (Hz)")
        plot_comp.setLabel("left", "Amplitudine (dB SPL)")
        plot_comp.setLogMode(x=True, y=False)
        plot_comp.setYRange(NIVEL_PODEA, NIVEL_PLAFON)
        plot_comp.showGrid(x=True, y=True, alpha=0.3)
        plot_comp.addLegend()

        max_plot_frequency = min(20000, nyquist)
        min_plot_frequency = min(20, max_plot_frequency / 10)
        plot_comp.setXRange(np.log10(min_plot_frequency), np.log10(max_plot_frequency))

        plot_comp.plot(
            freqs_disp, db_orig_disp,
            pen=pg.mkPen(COL_ORANGE, width=1.5),
            name="Spectru Z (nefiltrat)",
        )
        plot_comp.plot(
            freqs_disp, db_filt_disp,
            pen=pg.mkPen(COL_CYAN, width=1.5),
            name=f"Spectru {TIP_PONDERARE}",
        )
        app2.exec_()