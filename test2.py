import warnings
import os
import sys
import time
import queue
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
    from pyqtgraph.Qt import QtCore, QtWidgets
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

# ~~~~~SELECTIE PONDERARE FRECVENTA (IEC 61672-1)~~~~~
# A-Weighting: ponderare standard pentru perceptia auditiva (majoritatea masuratorilor de zgomot)
# C-Weighting: ponderare aproape plata, folosita pentru niveluri de varf / semnale puternice
# Z-Weighting: fara ponderare (raspuns "zero"/flat), semnalul ramane nemodificat
#   -> conform Anexei E, Ec. (E.9): Z(f) = 0 dB pentru orice frecventa. Semnalul "nefiltrat"
#      din acest program ESTE de fapt semnalul Z-weighted (asa il numim in etichetele Leq).
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

# ~~~~~SELECTIE ANALIZA PE BENZI DE FRECVENTA (IEC 61260-1)~~~~~
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

# ~~~~~CALIBRARE SPL (IEC 61672-1, 5.2 - Adjustments at the calibration check frequency)~~~~~
# Constanta de calibrare converteste nivelul relativ (dBFS) in nivel absolut (dB SPL).
# Se determina practic aplicand un calibrator acustic (ex: 94 dB SPL la 1 kHz) la microfon,
# citind valoarea dBFS afisata de program, si calculand: CALIBRARE_DB = SPL_cunoscut - dBFS_citit.
# Pentru moment, valoarea este introdusa manual de la tastatura.
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

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE = 48000
EPSILON = 1e-12
AUDIO_NORM = None
data_queue = queue.Queue()
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
# Polii/zerourile de mai jos corespund valorilor exacte din formulele analitice
# IEC 61672-1 Anexa E (Ec. E.1 - E.8): f1=20.598997, f2=107.65265, f3=737.86223,
# f4=12194.217 Hz, cu normalizarile A1000/C1000 pentru castig 0 dB la 1 kHz.

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
else:  # Z-Weighting: fara filtrare, raspuns flat (Anexa E, Ec. E.9: Z(f) = 0 dB)
    sos_ponderare = None
    zi_ponderare = None

def filtreaza_block(chunk):
    """Aplica ponderarea de frecventa selectata (A, C sau Z) semnalului brut."""
    global zi_ponderare
    if sos_ponderare is None:
        return chunk.copy()
    chunk_ponderat, zi_ponderare = sosfilt(sos_ponderare, chunk, zi=zi_ponderare)
    return chunk_ponderat

########################################################
### PONDERARE EXPONENTIALA IN TIMP (IEC 61672-1, 3.6) ###
########################################################
# Ecuatia (1) / Figura 1 din IEC 61672-1 defineste nivelul ponderat in timp ca:
#   L(t) = 10*log10[ (1/tau) * integrala{-inf,t} p^2(ksi) * e^{-(t-ksi)/tau} dksi / p0^2 ]
# adica: ridicare la patrat -> filtru trece-jos cu UN SINGUR POL REAL la -1/tau
# (ponderare exponentiala) -> log10 -> afisare in dB.
# Discret, acest filtru de ordinul 1 este exact echivalent cu un IIR:
#   y[n] = alpha*x[n] + (1-alpha)*y[n-1],  alpha = 1 - exp(-1/(tau*fs))
# aplicat PE FIECARE ESANTION (T = 1/fs), cu stare persistenta intre blocuri audio
# (nu se reseteaza niciodata cat timp masuratoarea e activa).
#
# tau_F = 0.125 s, tau_S = 1 s (5.8.1). Pentru "Peak" nu se aplica aceasta ponderare,
# ci se ia direct esantionul de varf absolut (3.8/3.9), vezi mai jos.

def creeaza_filtru_timp(tau, fs):
    """Returneaza coeficientii (b, a) ai filtrului IIR de ordinul 1 echivalent cu
    ponderarea exponentiala in timp de constanta tau, esantionata la fs."""
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
    TAU_TIMP = 0.125  # nefolosit efectiv in modul Peak, dar pastram o valoare valida

b_timp, a_timp = creeaza_filtru_timp(TAU_TIMP, SAMPLE_RATE)
zi_nivel_raw = [0.0]        # stare filtru IIR pt nivelul pe semnalul Z (nefiltrat)
zi_nivel_filt = [0.0]       # stare filtru IIR pt nivelul pe semnalul ponderat A/C/Z

def nivel_ponderat_in_timp(chunk, b, a, zi):
    """Aplica exact Ecuatia (1)/Figura 1 din IEC 61672-1: patrat -> IIR ordin 1 (pol -1/tau)
    -> (log10 se aplica separat, in dB, de catre apelant). Returneaza (ms_instantaneu, zi_nou)."""
    ms, zi_nou = lfilter(b, a, np.square(chunk), zi=zi)
    return ms, zi_nou

def db_din_ms(valoare_ms):
    return float(np.clip(10.0 * np.log10(valoare_ms + EPSILON) + CALIBRARE_DB, NIVEL_PODEA, NIVEL_PLAFON))

########################################################
############ Leq (time-averaged sound level) ###########
########################################################
# Conform IEC 61672-1, 3.10, Nota 3: "In principle, time weighting is not involved in
# a determination of time-averaged sound level" -> Leq e o MEDIE LINIARA simpla a
# patratului semnalului pe intervalul T, NU un filtru exponential.
#   L_eq,T = 10*log10[ (1/T) * integrala{t-T,t} p^2(ksi) dksi / p0^2 ]
# Calculam Leq atat pe semnalul nefiltrat (care, prin definitie, e Z-weighted -
# Anexa E, Ec. E.9) cat si pe cel ponderat (A/C, dupa selectia utilizatorului).
# T = intervalul scurs de la ultimul reset (sau de la pornirea masuratorii).

leq_state = {
    "suma_patrate_raw": 0.0,
    "suma_patrate_filt": 0.0,
    "n_esantioane": 0,
}

def actualizeaza_leq(chunk_raw, chunk_ponderat):
    leq_state["suma_patrate_raw"] += float(np.sum(np.square(chunk_raw)))
    leq_state["suma_patrate_filt"] += float(np.sum(np.square(chunk_ponderat)))
    leq_state["n_esantioane"] += len(chunk_raw)

def calculeaza_leq():
    """Returneaza (L_Zeq, L_Xeq) unde X e ponderarea selectata (A/C/Z)."""
    n = leq_state["n_esantioane"]
    if n == 0:
        return NIVEL_PODEA, NIVEL_PODEA
    mp_raw = leq_state["suma_patrate_raw"] / n
    mp_filt = leq_state["suma_patrate_filt"] / n
    l_zeq = db_din_ms(mp_raw)
    l_xeq = db_din_ms(mp_filt)
    return l_zeq, l_xeq

def reseteaza_leq():
    leq_state["suma_patrate_raw"] = 0.0
    leq_state["suma_patrate_filt"] = 0.0
    leq_state["n_esantioane"] = 0

########################################################
##### BANC DE FILTRE PE BENZI DE OCTAVA (IEC 61260-1) ###
########################################################
# Seria de baza 10 (preferata de IEC 61260-1): raport de octava G = 10^(3/10)
#
# Frecventa centrala exacta (exact mid-band frequency), conform IEC 61260-1 §5.4:
#   - cand numitorul lui 1/b este IMPAR (b=1, b=3 -> octave, terte), §5.4.1:
#         f_c = f_ref * G^(x/b)                       [Formula (2)]
#   - cand numitorul lui 1/b este PAR (b=6, b=12 -> sesimi, doisprezecimi), §5.4.2:
#         f_c = f_ref * G^((2x+1)/(2b))                [Formula (3)]
#
# Limite de banda (independente de paritatea lui b, §5.6):
#     f_low = f_c * G^(-1/2b) ,  f_high = f_c * G^(1/2b)
# unde b = numarul de benzi pe octava (1, 3, 6 sau 12), f_ref = 1000 Hz

OCTAVE_G = 10 ** (3.0 / 10.0)
FREQ_REF = 1000.0

def formateaza_frecventa(f):
    if f >= 1000:
        return f"{f/1000:.3g}k"
    return f"{f:.3g}"

def frecventa_centrala_exacta(x, b):
    """Calculeaza frecventa centrala exacta conform IEC 61260-1 §5.4.

    - Formula (2), §5.4.1, pentru numitor impar al lui 1/b (b=1, b=3)
    - Formula (3), §5.4.2, pentru numitor par al lui 1/b (b=6, b=12)
    """
    if b % 2 == 1:
        return FREQ_REF * OCTAVE_G ** (x / b)
    else:
        return FREQ_REF * OCTAVE_G ** ((2 * x + 1) / (2 * b))

def design_octave_bands(fs, b, fmin=20.0, fmax=20000.0):
    """Proiecteaza un banc de filtre trece-banda Butterworth (sos) pe benzi de 1/b octava,
    conform frecventelor centrale/limitelor standardizate IEC 61260-1 (seria de baza 10)."""
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
        # filtru IIR de ordinul 1 pentru ponderarea exponentiala in timp (acelasi tau
        # ca la nivelul principal), aplicat pe patratul semnalului filtrat pe banda -
        # inlocuieste vechea actualizare aproximativa "per bloc".
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
    """Filtreaza semnalul ponderat prin fiecare banda de octava, apoi aplica exact
    aceeasi ponderare exponentiala in timp (Ec. 1, IEC 61672-1) ca la nivelul principal,
    per esantion, cu stare persistenta intre blocuri. Se aplica si CALIBRARE_DB."""
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
# IMPORTANT: fereastra Hanning si bufferul circular de mai jos servesc STRICT pentru
# afisarea spectrului FFT. Nivelul (dB) NU se mai calculeaza din acest buffer -
# se calculeaza continuu, prin filtrul IIR de mai sus, ca sa respecte Ecuatia (1).

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

########################################################
#############PROCESARE AUDIO SI ANALIZA#################

def actualizeaza_ring_buffer(buffer, chunk):
    frames = len(chunk)
    if frames >= len(buffer):
        buffer[:] = chunk[-len(buffer):]
    else:
        buffer[:] = np.roll(buffer, -frames)
        buffer[-frames:] = chunk

def calculeaza_fft_pentru_afisare(buffer):
    """Doar pentru afisarea vizuala a spectrului - foloseste fereastra Hanning,
    NU influenteaza nivelul (dB) afisat, care se calculeaza separat prin filtrul IIR."""
    windowed_signal = buffer * hanning_window
    fft_raw = np.abs(np.fft.rfft(windowed_signal))
    fft_norm = fft_raw / (WINDOW_SIZE_FFT / 2.0)
    fft_db = 20 * np.log10(fft_norm + EPSILON) + CALIBRARE_DB
    fft_db = np.clip(fft_db, NIVEL_PODEA, NIVEL_PLAFON)
    return fft_db

def calculeaza_peak_db(buffer):
    """Peak sound level, conform IEC 61672-1, 3.8/3.9: cel mai mare esantion (in valoare
    absoluta) dintr-un interval - NU o medie RMS. p0 implicit 1.0 (semnal normalizat)."""
    peak = np.max(np.abs(buffer))
    val = 20 * np.log10(peak + EPSILON) + CALIBRARE_DB
    return float(np.clip(val, NIVEL_PODEA, NIVEL_PLAFON))

peak_hold_state = {
    "raw_db": NIVEL_PODEA,
    "filt_db": NIVEL_PODEA,
}

def get_peak_hold_display(cheie, valoare_curenta):
    val_key = f"{cheie}_db"
    if valoare_curenta > peak_hold_state[val_key]:
        peak_hold_state[val_key] = valoare_curenta
    return peak_hold_state[val_key]

def reseteaza_peak_hold():
    peak_hold_state["raw_db"] = NIVEL_PODEA
    peak_hold_state["filt_db"] = NIVEL_PODEA

def trimite_date_live(chunk, chunk_ponderat):
    global zi_nivel_raw, zi_nivel_filt
    current_time = play_pointer / SAMPLE_RATE

    # Leq - actualizat mereu, indiferent de mod (medie liniara, fara ponderare exponentiala)
    actualizeaza_leq(chunk, chunk_ponderat)
    l_zeq, l_xeq = calculeaza_leq()

    if PEAK_MODE:
        db_raw = calculeaza_peak_db(chunk)
        db_filtered = calculeaza_peak_db(chunk_ponderat)
        data_queue.put((current_time, db_raw, db_filtered, None, None, None, l_zeq, l_xeq))
    else:
        # Nivel ponderat in timp (F/S), conform Ecuatiei (1) - filtru IIR continuu,
        # stare persistenta intre blocuri (NU se reseteaza).
        ms_raw, zi_nivel_raw = nivel_ponderat_in_timp(chunk, b_timp, a_timp, zi_nivel_raw)
        ms_filt, zi_nivel_filt = nivel_ponderat_in_timp(chunk_ponderat, b_timp, a_timp, zi_nivel_filt)
        db_raw = db_din_ms(ms_raw[-1])
        db_filtered = db_din_ms(ms_filt[-1])

        # buffer separat, doar pentru afisarea FFT
        actualizeaza_ring_buffer(live_ring_buffer_raw_fft, chunk)
        actualizeaza_ring_buffer(live_ring_buffer_filtered_fft, chunk_ponderat)
        fft_raw = calculeaza_fft_pentru_afisare(live_ring_buffer_raw_fft) if plot_fft is not None else None
        fft_filtered = calculeaza_fft_pentru_afisare(live_ring_buffer_filtered_fft) if plot_fft is not None else None

        niveluri_benzi = proceseaza_benzi(chunk_ponderat) if plot_bands is not None else None
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
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_ponderat.copy())
    trimite_date_live(chunk, chunk_ponderat)
    if valid_frames < frames:
        raise sd.CallbackStop()

def record_callback(indata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
    chunk = indata[:, 0].astype(np.float64, copy=True)
    chunk_ponderat = filtreaza_block(chunk)
    play_pointer += len(chunk)
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_ponderat.copy())
    trimite_date_live(chunk, chunk_ponderat)

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
    # Fereastra hibrida: Grafice in stanga, Meter separat in dreapta
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Analiza DSP live{titlu_context_str} - Mod: {MODE}")
    win.resize(1280, 960)
    win.setStyleSheet("background-color: black;")

    layout_main = QtWidgets.QHBoxLayout(win)

    # Layout pyqtgraph pentru grafice
    win_graphics = pg.GraphicsLayoutWidget()
    layout_main.addWidget(win_graphics, stretch=4)

    # Panou separat pentru meter-ul de nivel
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
        curve_db_raw = plot_db.plot(pen=pg.mkPen(COL_ORANGE, width=2), name=f"Z (nefiltrat) - nivel ({MODE}, IEC 61672-1 Ec.1)")
        curve_db_filtered = plot_db.plot(pen=pg.mkPen(COL_PURPLE, width=2), name=f"{TIP_PONDERARE} - nivel ({MODE}, IEC 61672-1 Ec.1)")

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

x_data = []
y_db_raw = []
y_db_filtered = []

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
            last_fft_raw = fft_raw
            last_fft_filtered = fft_filtered
            last_niveluri_benzi = niveluri_benzi
            last_leq_raw = leq_raw
            last_leq_filt = leq_filt
            updated = True
        except queue.Empty:
            break

    if not updated:
        if stream is not None and not stream.active:
            running["active"] = False
            app.quit()
        return

    # Actualizam panoul de meter separat
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
            curve_db_raw.setData(x_data, y_db_raw)
            curve_db_filtered.setData(x_data, y_db_filtered)

        if plot_fft is not None and last_fft_raw is not None:
            curve_fft_raw.setData(fft_frequencies_disp, last_fft_raw)
            curve_fft_filtered.setData(fft_frequencies_disp, last_fft_filtered)

        if plot_bands is not None and last_niveluri_benzi is not None:
            bar_item.setOpts(height=last_niveluri_benzi - NIVEL_PODEA)

    if stream is not None and not stream.active:
        running["active"] = False
        app.quit()

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(16)  # ~60 fps

try:
    if SURSA_OPT == "1":
        print(f"se reda audio ponderat {TIP_PONDERARE} si se afiseaza comparatia live + benzi de 1/{FRACTIE_OCTAVA} octava")
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=playback_callback,
        )
    else:
        print(f"se analizeaza microfonul cu ponderare {TIP_PONDERARE} si se afiseaza comparatia live + benzi de 1/{FRACTIE_OCTAVA} octava")
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=record_callback,
        )

    with stream:
        app.exec_()

except KeyboardInterrupt:
    print("\nmonitorizare oprita de utilizator.")

finally:
    if HAS_SOUNDDEVICE:
        sd.stop()
    print("\nprocesare finalizata")

    l_zeq_final, l_xeq_final = calculeaza_leq()
    print(f"Leq final (durata masurata): L_Zeq = {l_zeq_final:.1f} dB | L_{SIMBOL_PONDERARE}eq = {l_xeq_final:.1f} dB")

    if semnal_nefiltrat_complet and not PEAK_MODE:
        semnal_complet_raw = np.concatenate(semnal_nefiltrat_complet)
        semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

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