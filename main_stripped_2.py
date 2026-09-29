import sys
import queue
import threading
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, bilinear_zpk, zpk2sos, lfilter, firwin2

try:
    import sounddevice as sd
except ImportError:
    print("Error loading sounddevice")
    sys.exit()
try:
    from pyqtgraph.Qt import QtCore, QtWidgets   # doar wrapper-ul Qt, fara grafice
except ImportError:
    print("Error loading pyqtgraph")
    sys.exit()

########################################################
################### SELECTIE DISPOZITIV ################
########################################################

print("\n\n~~~~~CONFIG AUDIO~~~~~")
input_indices = []
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        print(f"[{i}] {dev['name']} (Input channels: {dev['max_input_channels']})")
        input_indices.append(i)

input_device_id = None
try:
    id_in = input("select device:\n>").strip()
    if id_in:
        id_in = int(id_in)
        if id_in in input_indices:
            input_device_id = id_in
        else:
            print("Invalid ID, using default")
except ValueError:
    print("Invalid value, using default")

########################################################
##################### SETARI ###########################
########################################################

print("~~~~~CORECTIE MICROFON~~~~~")
APLICA_CORECTIE_MICROFON = input(
    "Aplici corectia de raspuns in frecventa a microfonului? (d/n) > "
).strip().lower() in ("d", "da", "y", "yes")

print("~~~~~CONFIG MODE~~~~~")
MODE_IN = input("Mode:\n1.Fast\n2.Slow\n>").strip().lower()
if MODE_IN in ("2", "slow"):
    MODE, TAU_TIMP = "Slow", 1.0
else:
    if MODE_IN not in ("1", "fast"):
        print("Mod invalid. Se foloseste Fast")
    MODE, TAU_TIMP = "Fast", 0.125

print("~~~~~SELECT WEIGHTING MODE~~~~~")
print("1 A-Weighting\n2 C-Weighting\n3 Z-Weighting (flat)")
opt = input("> ").strip().lower()
if opt in ("2", "c", "c-weighting", "c-weight"):
    TIP_PONDERARE, SIMBOL_PONDERARE = "C-Weighting", "C"
elif opt in ("3", "z", "z-weighting", "z-weight"):
    TIP_PONDERARE, SIMBOL_PONDERARE = "Z-Weighting", "Z"
else:
    if opt not in ("1", "a", "a-weighting", "a-weight"):
        print("Invalid option. Using A-Weighting")
    TIP_PONDERARE, SIMBOL_PONDERARE = "A-Weighting", "A"

print("\n~~~~~CALIBRARE SPL~~~~~")
print("Constanta = SPL_cunoscut - dBFS_masurat (ex: calibrator 94 dB la 1 kHz).")
try:
    CALIBRARE_DB = float(input("Constanta de calibrare (dB) > ").strip().replace(",", "."))
except ValueError:
    print("Valoare invalida. Se foloseste 0.0 dB")
    CALIBRARE_DB = 0.0

SAMPLE_RATE = 48000
BLOCKSIZE = 256
LATENCY = "low"
EPSILON = 1e-12
GUI_REFRESH_MS = 50

NIVEL_PODEA = -120.0 + CALIBRARE_DB
NIVEL_PLAFON = 20.0 + CALIBRARE_DB

raw_queue = queue.Queue(maxsize=32)   # audio thread -> processing thread
stop_event = threading.Event()

########################################################
################ STARE PARTAJATA (lock) ################
########################################################

state_lock = threading.Lock()
state = {
    "sum_raw": 0.0, "sum_filt": 0.0, "n": 0,          # pentru Leq
    "peak_raw": NIVEL_PODEA, "peak_filt": NIVEL_PODEA, # peak hold (dB)
    "db_raw": NIVEL_PODEA, "db_filt": NIVEL_PODEA,     # nivelul curent (dB)
}

def db_din_ms(ms):
    return float(np.clip(10.0 * np.log10(ms + EPSILON) + CALIBRARE_DB, NIVEL_PODEA, NIVEL_PLAFON))

def reseteaza_peak():
    with state_lock:
        state["peak_raw"] = NIVEL_PODEA
        state["peak_filt"] = NIVEL_PODEA

def reseteaza_leq():
    with state_lock:
        state["sum_raw"] = state["sum_filt"] = 0.0
        state["n"] = 0

def actualizeaza_calibrare(valoare):
    global CALIBRARE_DB, NIVEL_PODEA, NIVEL_PLAFON
    CALIBRARE_DB = valoare
    NIVEL_PODEA = -120.0 + valoare
    NIVEL_PLAFON = 20.0 + valoare
    reseteaza_peak()
    reseteaza_leq()

########################################################
############ FILTRE DE PONDERARE (IEC 61672) ###########
########################################################

def prewarp(f, fs):
    return (fs / np.pi) * np.tan(np.pi * f / fs)

def filtru_a(fs):
    f1, f2, f3, f4 = [prewarp(f, fs) for f in (20.598997, 107.65265, 737.86223, 12194.217)]
    p1, p2, p3, p4 = [-2 * np.pi * f for f in (f1, f2, f3, f4)]
    k = (2 * np.pi * f4) ** 2 * 10 ** (1.9997 / 20)
    zd, pd, gd = bilinear_zpk([0, 0, 0, 0], [p1, p1, p2, p3, p4, p4], k, fs)
    return zpk2sos(zd, pd, gd)

def filtru_c(fs):
    f1, f4 = prewarp(20.598997, fs), prewarp(12194.217, fs)
    p1, p4 = -2 * np.pi * f1, -2 * np.pi * f4
    k = (2 * np.pi * f4) ** 2 * 10 ** (0.0619 / 20)
    zd, pd, gd = bilinear_zpk([0, 0], [p1, p1, p4, p4], k, fs)
    return zpk2sos(zd, pd, gd)

if TIP_PONDERARE == "A-Weighting":
    sos_ponderare = filtru_a(SAMPLE_RATE)
elif TIP_PONDERARE == "C-Weighting":
    sos_ponderare = filtru_c(SAMPLE_RATE)
else:
    sos_ponderare = None
zi_ponderare = sosfilt_zi(sos_ponderare) * 0.0 if sos_ponderare is not None else None

########################################################
############## CORECTIE CURBA MICROFON (FIR) ###########
########################################################

CALE_CALIBRARE_MICROFON = "microphone_profiles/35Y228_cal_0degree.txt"

def construieste_filtru_corectie(fs, freq_cal, dev_cal_db, numtaps=513):
    freqs = np.clip(np.concatenate(([0.0], freq_cal / (fs / 2.0), [1.0])), 0.0, 1.0)
    freqs, idx = np.unique(freqs, return_index=True)
    g = 10 ** (-dev_cal_db / 20.0)
    g = np.concatenate(([g[0]], g, [g[-1]]))[idx]
    return firwin2(numtaps, freqs, g)

fir_mic = None
zi_mic = None
if APLICA_CORECTIE_MICROFON:
    try:
        date = np.loadtxt(CALE_CALIBRARE_MICROFON)
        fir_mic = construieste_filtru_corectie(SAMPLE_RATE, date[:, 0], date[:, 1])
        zi_mic = np.zeros(len(fir_mic) - 1)
        print(f"Corectie microfon incarcata ({len(fir_mic)} taps)")
    except (OSError, ValueError) as e:
        print(f"Curba de calibrare nu a putut fi incarcata: {e}")
        fir_mic = None

########################################################
################### CALLBACK AUDIO #####################
########################################################

def record_callback(indata, frames, time_info, status):
    global zi_mic, zi_ponderare
    chunk = indata[:, 0].astype(np.float32, copy=True)
    if fir_mic is not None:
        chunk, zi_mic = lfilter(fir_mic, [1.0], chunk, zi=zi_mic)
        chunk = chunk.astype(np.float32, copy=False)
    if sos_ponderare is not None:
        chunk_pond, zi_ponderare = sosfilt(sos_ponderare, chunk, zi=zi_ponderare)
    else:
        chunk_pond = chunk
    try:
        raw_queue.put_nowait((chunk, chunk_pond))
    except queue.Full:
        pass

########################################################
################ THREAD DE PROCESARE ###################
########################################################

alpha = 1.0 - np.exp(-1.0 / (TAU_TIMP * SAMPLE_RATE))
b_timp = [alpha]
a_timp = [1.0, -(1.0 - alpha)]
zi_raw = [0.0]
zi_filt = [0.0]

def processing_loop():
    global zi_raw, zi_filt
    while True:
        try:
            chunk, chunk_pond = raw_queue.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        sp_raw = np.square(chunk)
        sp_filt = np.square(chunk_pond)
        ms_raw, zi_raw = lfilter(b_timp, a_timp, sp_raw, zi=zi_raw)
        ms_filt, zi_filt = lfilter(b_timp, a_timp, sp_filt, zi=zi_filt)
        db_raw = db_din_ms(ms_raw[-1])
        db_filt = db_din_ms(ms_filt[-1])

        with state_lock:
            state["sum_raw"] += float(sp_raw.sum())
            state["sum_filt"] += float(sp_filt.sum())
            state["n"] += len(chunk)
            state["db_raw"] = db_raw
            state["db_filt"] = db_filt
            if db_raw > state["peak_raw"]:
                state["peak_raw"] = db_raw
            if db_filt > state["peak_filt"]:
                state["peak_filt"] = db_filt

def citeste_stare():
    with state_lock:
        n = state["n"]
        if n:
            leq_raw = db_din_ms(state["sum_raw"] / n)
            leq_filt = db_din_ms(state["sum_filt"] / n)
        else:
            leq_raw = leq_filt = NIVEL_PODEA
        return (state["db_raw"], state["db_filt"],
                state["peak_raw"], state["peak_filt"], leq_raw, leq_filt)

########################################################
################# INTERFATA GRAFICA ####################
########################################################

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

STIL_BUTON = (
    "QPushButton { background-color: #333; color: white; padding: 6px; font-size: 11px; }"
    "QPushButton:hover { background-color: #555; }"
    "QPushButton:pressed { background-color: #777; }"
)
STIL_CAMP_NORMAL = "QLineEdit { background-color: #222; color: white; border: 1px solid #555; padding: 4px; }"
STIL_CAMP_EROARE = "QLineEdit { background-color: #222; color: white; border: 1px solid #e74c3c; padding: 4px; }"
STIL_BARA = (
    "QProgressBar { background-color: #222; border: 1px solid #555; }"
    "QProgressBar::chunk { background-color: %s; }"
)

def culoare_pentru_nivel(db):
    if db > -6.0 + CALIBRARE_DB:
        return "#e74c3c"
    if db > -18.0 + CALIBRARE_DB:
        return "#f1c40f"
    return "#2ecc71"

def eticheta(text, stil):
    e = QtWidgets.QLabel(text)
    e.setStyleSheet(stil)
    e.setAlignment(QtCore.Qt.AlignCenter)
    return e

class Coloana:
    def __init__(self, nume):
        self.layout = QtWidgets.QVBoxLayout()
        self.bara = QtWidgets.QProgressBar()
        self.bara.setOrientation(QtCore.Qt.Vertical)
        self.bara.setRange(0, 1200)
        self.bara.setTextVisible(False)
        self.bara.setFixedWidth(50)
        self.bara.setStyleSheet(STIL_BARA % "#2ecc71")
        self.culoare = "#2ecc71"
        self.e_val = eticheta(f"{NIVEL_PODEA:.1f} dB", "color: white; font-size: 16px; font-weight: bold;")
        self.e_peak = eticheta(f"Peak: {NIVEL_PODEA:.1f} dB", "color: #f1c40f; font-size: 12px;")
        self.e_leq = eticheta(f"Leq: {NIVEL_PODEA:.1f} dB", "color: #3498db; font-size: 12px;")
        self.layout.addWidget(eticheta(nume, "color: white; font-size: 13px; font-weight: bold;"))
        self.layout.addWidget(self.bara, alignment=QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.e_val)
        self.layout.addWidget(self.e_peak)
        self.layout.addWidget(self.e_leq)

    def actualizeaza(self, db, peak, leq):
        interval = NIVEL_PLAFON - NIVEL_PODEA
        self.bara.setValue(int(np.clip((db - NIVEL_PODEA) / interval * 1200, 0, 1200)))
        culoare = culoare_pentru_nivel(db)
        if culoare != self.culoare:            # stylesheet doar cand se schimba culoarea
            self.culoare = culoare
            self.bara.setStyleSheet(STIL_BARA % culoare)
        self.e_val.setText(f"{db:6.1f} dB")
        self.e_peak.setText(f"Peak: {peak:6.1f} dB")
        self.e_leq.setText(f"Leq: {leq:6.1f} dB")

win = QtWidgets.QWidget()
win.setWindowTitle(f"Sonometru [{TIP_PONDERARE} | {MODE}]")
win.resize(320, 620)
win.setStyleSheet("background-color: black;")
layout = QtWidgets.QVBoxLayout(win)

col_raw = Coloana("Z (nefiltrat)")
col_filt = Coloana(TIP_PONDERARE)
layout_bare = QtWidgets.QHBoxLayout()
layout_bare.addLayout(col_raw.layout)
layout_bare.addLayout(col_filt.layout)
layout.addLayout(layout_bare)

# --- calibrare ---
e_calib = eticheta(f"In uz: {CALIBRARE_DB:+.2f} dB", "color: #2ecc71; font-size: 12px; font-weight: bold;")
camp_calib = QtWidgets.QLineEdit()
camp_calib.setPlaceholderText("noua constanta (dB)")
camp_calib.setStyleSheet(STIL_CAMP_NORMAL)
camp_calib.setAlignment(QtCore.Qt.AlignCenter)
buton_calib = QtWidgets.QPushButton("Aplica")
buton_calib.setStyleSheet(STIL_BUTON)

def handler_calibrare():
    text = camp_calib.text().strip().replace(",", ".")
    if not text:
        return
    try:
        valoare = float(text)
    except ValueError:
        camp_calib.setStyleSheet(STIL_CAMP_EROARE)
        return
    actualizeaza_calibrare(valoare)
    e_calib.setText(f"In uz: {CALIBRARE_DB:+.2f} dB")
    camp_calib.clear()
    camp_calib.setStyleSheet(STIL_CAMP_NORMAL)

buton_calib.clicked.connect(handler_calibrare)
camp_calib.returnPressed.connect(handler_calibrare)

buton_peak = QtWidgets.QPushButton("Reset Peak")
buton_peak.setStyleSheet(STIL_BUTON)
buton_peak.clicked.connect(reseteaza_peak)
buton_leq = QtWidgets.QPushButton("Reset Leq")
buton_leq.setStyleSheet(STIL_BUTON)
buton_leq.clicked.connect(reseteaza_leq)

layout.addWidget(eticheta("Calibrare SPL", "color: white; font-size: 12px; font-weight: bold;"))
layout.addWidget(e_calib)
layout.addWidget(camp_calib)
layout.addWidget(buton_calib)
layout.addWidget(buton_peak)
layout.addWidget(buton_leq)
win.show()

def update_gui():
    db_raw, db_filt, pk_raw, pk_filt, leq_raw, leq_filt = citeste_stare()
    col_raw.actualizeaza(db_raw, pk_raw, leq_raw)
    col_filt.actualizeaza(db_filt, pk_filt, leq_filt)

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(GUI_REFRESH_MS)

########################################################
################## PORNIREA STREAMULUI #################
########################################################

processing_thread = threading.Thread(target=processing_loop, daemon=True)
processing_thread.start()

try:
    print(f"Se analizeaza microfonul | {MODE} | {TIP_PONDERARE}")
    with sd.InputStream(device=input_device_id, samplerate=SAMPLE_RATE, channels=1,
                        blocksize=BLOCKSIZE, latency=LATENCY, callback=record_callback):
        app.exec_()
except KeyboardInterrupt:
    print("\nMonitorizare oprita de utilizator.")
finally:
    stop_event.set()
    processing_thread.join(timeout=2.0)
    _, _, _, _, leq_z, leq_x = citeste_stare()
    print(f"Leq final: L_Zeq = {leq_z:.1f} dB | L_{SIMBOL_PONDERARE}eq = {leq_x:.1f} dB")
    print(f"Constanta de calibrare la final: {CALIBRARE_DB:+.2f} dB")