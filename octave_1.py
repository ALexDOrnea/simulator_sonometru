import warnings
import os
import sys
import queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfilt_zi, bilinear_zpk, zpk2sos, butter

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.io.wavfile")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    print("Sounddevice nu a fost initializat. incearca sa il instalezi")
    sys.exit()

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("pyqtgraph nu a fost gasit. instaleaza cu: pip install pyqtgraph pyqt5")
    sys.exit()

###########################################
############Selectie in/out################

input_device_id = None
output_device_id = None

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

print("\n~~~~~SELECTIE FILTRU PONDERARE~~~~~")
print("1 A-Weighting (IEC 61672)")
print("2 C-Weighting (IEC 61672)")
FILTRU_OPT = input("> ").strip().lower()

if FILTRU_OPT in ("1", "a", "a-weighting", "a-weight"):
    TIP_FILTRU = "A-Weighting"
elif FILTRU_OPT in ("2", "c", "c-weighting", "c-weight"):
    TIP_FILTRU = "C-Weighting"
else:
    print("Invalid option. using A-Weighting")
    TIP_FILTRU = "A-Weighting"

print("\n~~~~~SELECTIE BANDA OCTAVA (SONOMETRU)~~~~~")
print("1  1/1 octava")
print("2  1/3 octava")
print("3  1/6 octava")
print("4  1/12 octava")
BANDA_OPT = input("> ").strip()
FRACTIE_OCTAVA = {"1": 1, "2": 3, "3": 6, "4": 12}.get(BANDA_OPT, 3)
print(f"Banda selectata: 1/{FRACTIE_OCTAVA} octava")

if PEAK_MODE:
    GRAFIC_OPT = None
    print("\n~~~~~MOD PEAK~~~~~")
    print("Se va afisa un live meter cu maximul in dB.")
else:
    print("\n~~~~~SELECTIE GRAFICE~~~~~")
    print("1 dB FS")
    print("2 FFT")
    print("3 All")
    GRAFIC_OPT = input("> ").strip()
    if GRAFIC_OPT not in ("1", "2", "3"):
        print("Invalid option.using all")
        GRAFIC_OPT = "3"

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE = 44100
EPSILON = 1e-12
AUDIO_NORM = None
data_queue = queue.Queue()
play_pointer = 0

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

###########################################
###########Filtre de ponderare#############

def get_a_weighting_filter(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = -2.000
    p1, p2, p3, p4 = -2*np.pi*f1, -2*np.pi*f2, -2*np.pi*f3, -2*np.pi*f4
    z = [0, 0, 0, 0]
    p = [p1, p1, p2, p3, p4, p4]
    k = (2 * np.pi * f4)**2 * (10**(A1000 / 20))
    zeros_d, poles_d, gain_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(zeros_d, poles_d, gain_d)

def get_c_weighting_filter(fs):
    f1, f4 = 20.598997, 12194.217
    C1000 = -0.062
    p1, p4 = -2*np.pi*f1, -2*np.pi*f4
    z = [0, 0]
    p = [p1, p1, p4, p4]
    k = (2 * np.pi * f4)**2 * (10**(C1000 / 20))
    zeros_d, poles_d, gain_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(zeros_d, poles_d, gain_d)

if TIP_FILTRU == "A-Weighting":
    sos_filter = get_a_weighting_filter(SAMPLE_RATE)
else:
    sos_filter = get_c_weighting_filter(SAMPLE_RATE)

zi_filter = sosfilt_zi(sos_filter) * 0.0

def filtreaza_block(chunk):
    global zi_filter
    chunk_filtrat, zi_filter = sosfilt(sos_filter, chunk, zi=zi_filter)
    return chunk_filtrat

###########################################
#########Generare benzi de octava##########

def genereaza_benzi_octava(fractie, f_min=20.0, f_max=20000.0):
    G = 10 ** (3.0 / 10.0)
    benzi = []
    x_min = int(np.floor(fractie * np.log(f_min / 1000.0) / np.log(G))) - 1
    x_max = int(np.ceil(fractie * np.log(f_max / 1000.0) / np.log(G))) + 1
    for x in range(x_min, x_max + 1):
        fc = 1000.0 * G ** (x / fractie)
        if f_min <= fc <= f_max:
            f_lo = fc * G ** (-1.0 / (2.0 * fractie))
            f_hi = fc * G ** (1.0 / (2.0 * fractie))
            benzi.append((fc, f_lo, f_hi))
    return benzi

F_MAX_BENZI = min(20000.0, nyquist * 0.97)
BENZI_OCTAVA = genereaza_benzi_octava(FRACTIE_OCTAVA, f_min=20.0, f_max=F_MAX_BENZI)
print(f"Numar benzi de octava: {len(BENZI_OCTAVA)}")

# Construim filtrele band-pass pentru fiecare banda
sos_benzi = []
zi_benzi = []
for fc, f_lo, f_hi in BENZI_OCTAVA:
    f_lo_n = np.clip(f_lo / nyquist, 1e-5, 0.9999)
    f_hi_n = np.clip(f_hi / nyquist, 1e-5, 0.9999)
    if f_lo_n >= f_hi_n:
        f_hi_n = min(f_lo_n + 0.001, 0.9999)
    try:
        sos_b = butter(4, [f_lo_n, f_hi_n], btype="bandpass", output="sos")
    except Exception:
        sos_b = butter(2, [f_lo_n, f_hi_n], btype="bandpass", output="sos")
    sos_benzi.append(sos_b)
    zi_benzi.append(sosfilt_zi(sos_b) * 0.0)

def calculeaza_niveluri_octava(buffer):
    """Calculeaza RMS dB pentru fiecare banda de octava din buffer."""
    niveluri = []
    for i, sos_b in enumerate(sos_benzi):
        filtrat, zi_benzi[i] = sosfilt(sos_b, buffer, zi=zi_benzi[i])
        rms = np.sqrt(np.mean(np.square(filtrat)))
        db = float(np.clip(20.0 * np.log10(rms + EPSILON), -120.0, 0.0))
        niveluri.append(db)
    return niveluri

###########################################
###########Window size & buffers###########

if MODE.lower() == "fast":
    WINDOW_SIZE = int(0.125 * SAMPLE_RATE)
elif MODE.lower() == "slow":
    WINDOW_SIZE = int(1.0 * SAMPLE_RATE)
elif MODE.lower() == "peak":
    WINDOW_SIZE = int(0.035 * SAMPLE_RATE)
else:
    print("not a mode. using fast")
    WINDOW_SIZE = int(0.125 * SAMPLE_RATE)

live_ring_buffer_raw = np.zeros(WINDOW_SIZE)
live_ring_buffer_filtered = np.zeros(WINDOW_SIZE)
hanning_window = np.hanning(WINDOW_SIZE)
fft_frequencies = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLE_RATE)
FFT_DISPLAY_STEP = max(1, len(fft_frequencies) // 2000)
fft_frequencies_disp = fft_frequencies[::FFT_DISPLAY_STEP]

semnal_nefiltrat_complet = []
semnal_filtrat_complet = []

########################################################
#############PROCESARE AUDIO SI ANALIZA#################

def actualizeaza_ring_buffer(buffer, chunk):
    frames = len(chunk)
    if frames >= WINDOW_SIZE:
        buffer[:] = chunk[-WINDOW_SIZE:]
    else:
        buffer[:] = np.roll(buffer, -frames)
        buffer[-frames:] = chunk

def calculeaza_db_fft(buffer):
    rms = np.sqrt(np.mean(np.square(buffer)))
    db = np.clip(20 * np.log10(rms + EPSILON), -120.0, 0.0)
    windowed_signal = buffer * hanning_window
    fft_raw = np.abs(np.fft.rfft(windowed_signal))
    fft_norm = fft_raw / (WINDOW_SIZE / 2.0)
    fft_db = np.clip(20 * np.log10(fft_norm + EPSILON), -120.0, 0.0)
    return db, fft_db

def calculeaza_peak_db(buffer):
    peak = np.max(np.abs(buffer))
    return float(np.clip(20 * np.log10(peak + EPSILON), -120.0, 0.0))

peak_hold_state = {"raw_db": -120.0, "filt_db": -120.0}
peak_holds_octava = [-120.0] * len(BENZI_OCTAVA)
peak_hold_total = -120.0

def get_peak_hold_display(cheie, valoare_curenta):
    val_key = f"{cheie}_db"
    if valoare_curenta > peak_hold_state[val_key]:
        peak_hold_state[val_key] = valoare_curenta
    return peak_hold_state[val_key]

def reseteaza_peak_hold():
    global peak_hold_total
    peak_hold_state["raw_db"] = -120.0
    peak_hold_state["filt_db"] = -120.0
    for i in range(len(peak_holds_octava)):
        peak_holds_octava[i] = -120.0
    peak_hold_total = -120.0

def trimite_date_live(chunk, chunk_filtrat):
    current_time = play_pointer / SAMPLE_RATE
    actualizeaza_ring_buffer(live_ring_buffer_raw, chunk)
    actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_filtrat)

    if PEAK_MODE:
        db_raw = calculeaza_peak_db(live_ring_buffer_raw)
        db_filtered = calculeaza_peak_db(live_ring_buffer_filtered)
        niveluri_octava = calculeaza_niveluri_octava(live_ring_buffer_filtered)
        data_queue.put((current_time, db_raw, db_filtered, None, None, niveluri_octava))
    else:
        db_raw, fft_raw = calculeaza_db_fft(live_ring_buffer_raw)
        db_filtered, fft_filtered = calculeaza_db_fft(live_ring_buffer_filtered)
        niveluri_octava = calculeaza_niveluri_octava(live_ring_buffer_filtered)
        data_queue.put((
            current_time,
            db_raw,
            db_filtered,
            fft_raw[::FFT_DISPLAY_STEP],
            fft_filtered[::FFT_DISPLAY_STEP],
            niveluri_octava,
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
    chunk_filtrat = filtreaza_block(chunk)
    outdata.fill(0)
    outdata[:valid_frames, 0] = chunk_filtrat
    play_pointer += valid_frames
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_filtrat.copy())
    trimite_date_live(chunk, chunk_filtrat)
    if valid_frames < frames:
        raise sd.CallbackStop()

def record_callback(indata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
    chunk = indata[:, 0].astype(np.float64, copy=True)
    chunk_filtrat = filtreaza_block(chunk)
    play_pointer += len(chunk)
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_filtrat.copy())
    trimite_date_live(chunk, chunk_filtrat)

########################################################
################ Interfata grafica (pyqtgraph) ##########
########################################################

pg.setConfigOptions(antialias=False, useOpenGL=True, background="k", foreground="w")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

COL_ORANGE = (255, 165, 0)
COL_PURPLE = (170, 90, 220)
COL_CYAN = (0, 220, 220)
COL_GRAY = (120, 120, 120)

def culoare_pentru_nivel(db):
    if db > -6.0:
        return "#e74c3c"
    elif db > -18.0:
        return "#f1c40f"
    else:
        return "#2ecc71"

def creeaza_coloana_meter(nume, latime=44):
    coloana = QtWidgets.QVBoxLayout()
    eticheta_nume = QtWidgets.QLabel(nume)
    eticheta_nume.setStyleSheet("color: white; font-size: 9px; font-weight: bold;")
    eticheta_nume.setAlignment(QtCore.Qt.AlignCenter)
    eticheta_nume.setWordWrap(True)

    bara = QtWidgets.QProgressBar()
    bara.setOrientation(QtCore.Qt.Vertical)
    bara.setRange(0, 1200)
    bara.setValue(0)
    bara.setTextVisible(False)
    bara.setFixedWidth(latime)
    bara.setStyleSheet(
        "QProgressBar { background-color: #222; border: 1px solid #555; }"
        "QProgressBar::chunk { background-color: #2ecc71; }"
    )

    eticheta_valoare = QtWidgets.QLabel("-120 dB")
    eticheta_valoare.setStyleSheet("color: white; font-size: 8px;")
    eticheta_valoare.setAlignment(QtCore.Qt.AlignCenter)

    eticheta_peak = QtWidgets.QLabel("Pk:-120")
    eticheta_peak.setStyleSheet("color: #f1c40f; font-size: 8px;")
    eticheta_peak.setAlignment(QtCore.Qt.AlignCenter)

    coloana.addWidget(eticheta_nume)
    coloana.addWidget(bara, alignment=QtCore.Qt.AlignHCenter)
    coloana.addWidget(eticheta_valoare)
    coloana.addWidget(eticheta_peak)
    return coloana, bara, eticheta_valoare, eticheta_peak

def actualizeaza_bara(bara, eticheta_valoare, eticheta_peak, valoare_db, valoare_peak_db):
    bara.setValue(int(np.clip((valoare_db + 120.0) * 10, 0, 1200)))
    bara.setStyleSheet(
        "QProgressBar { background-color: #222; border: 1px solid #555; }"
        f"QProgressBar::chunk {{ background-color: {culoare_pentru_nivel(valoare_db)}; }}"
    )
    eticheta_valoare.setText(f"{valoare_db:.1f}")
    eticheta_peak.setText(f"Pk:{valoare_peak_db:.1f}")

###############################################
########### Fereastra 1 - Grafice principale ###
###############################################

plot_db = None
plot_fft = None
curve_db_raw = curve_db_filtered = None
curve_fft_raw = curve_fft_filtered = None

if PEAK_MODE:
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Live Peak Meter - {TIP_FILTRU}")
    win.resize(360, 560)
    win.setStyleSheet("background-color: black;")

    layout_vertical = QtWidgets.QVBoxLayout(win)
    layout_principal = QtWidgets.QHBoxLayout()

    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw = creeaza_coloana_meter("Nefiltrat", 50)
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt = creeaza_coloana_meter(
        f"Filtrat\n({TIP_FILTRU})", 50)

    layout_principal.addLayout(coloana_raw)
    layout_principal.addLayout(coloana_filt)

    buton_reset = QtWidgets.QPushButton("Reset Peak")
    buton_reset.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 8px; font-size: 13px; }"
        "QPushButton:hover { background-color: #555; }"
    )
    buton_reset.clicked.connect(reseteaza_peak_hold)

    layout_vertical.addLayout(layout_principal)
    layout_vertical.addWidget(buton_reset)
    win.show()

else:
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Analiza DSP live - {TIP_FILTRU} - Mod: {MODE}")
    win.resize(1280, 800)
    win.setStyleSheet("background-color: black;")

    layout_main = QtWidgets.QHBoxLayout(win)

    win_graphics = pg.GraphicsLayoutWidget()
    layout_main.addWidget(win_graphics, stretch=4)

    panel_meter = QtWidgets.QWidget()
    layout_meter = QtWidgets.QVBoxLayout(panel_meter)

    layout_bars = QtWidgets.QHBoxLayout()
    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw = creeaza_coloana_meter("Nefiltrat", 50)
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt = creeaza_coloana_meter(
        f"Filtrat\n({TIP_FILTRU})", 50)

    layout_bars.addLayout(coloana_raw)
    layout_bars.addLayout(coloana_filt)

    buton_reset = QtWidgets.QPushButton("Reset Peak")
    buton_reset.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 6px; font-size: 11px; }"
        "QPushButton:hover { background-color: #555; }"
    )
    buton_reset.clicked.connect(reseteaza_peak_hold)

    layout_meter.addLayout(layout_bars)
    layout_meter.addWidget(buton_reset)
    layout_main.addWidget(panel_meter, stretch=1)

    win.show()

    if GRAFIC_OPT in ("1", "3"):
        plot_db = win_graphics.addPlot(title=f"Nivel live: nefiltrat vs {TIP_FILTRU}")
        plot_db.setLabel("bottom", "Timp (s)")
        plot_db.setLabel("left", "Nivel (dB FS)")
        plot_db.setYRange(-120, 0)
        plot_db.showGrid(x=True, y=True, alpha=0.3)
        plot_db.addLegend()
        curve_db_raw = plot_db.plot(pen=pg.mkPen(COL_ORANGE, width=2),
                                     name=f"Nefiltrat ({MODE})")
        curve_db_filtered = plot_db.plot(pen=pg.mkPen(COL_PURPLE, width=2),
                                          name=f"Filtrat ({TIP_FILTRU}) ({MODE})")
        if SURSA_OPT == "1":
            plot_db.setXRange(0, len(AUDIO_NORM) / SAMPLE_RATE)
        else:
            plot_db.setXRange(0, 10)
        if GRAFIC_OPT == "3":
            win_graphics.nextRow()

    if GRAFIC_OPT in ("2", "3"):
        plot_fft = win_graphics.addPlot(title=f"FFT in timp real: nefiltrat vs {TIP_FILTRU}")
        plot_fft.setLabel("bottom", "Frecventa (Hz)")
        plot_fft.setLabel("left", "Amplitudine (dB FS)")
        plot_fft.setLogMode(x=True, y=False)
        plot_fft.setYRange(-120, 0)
        plot_fft.showGrid(x=True, y=True, alpha=0.3)
        plot_fft.addLegend()
        max_plot_frequency = min(20000, nyquist)
        min_plot_frequency = min(20, max_plot_frequency / 10)
        plot_fft.setXRange(np.log10(min_plot_frequency), np.log10(max_plot_frequency))
        curve_fft_raw = plot_fft.plot(pen=pg.mkPen(COL_ORANGE, width=1.5), name="FFT nefiltrat")
        curve_fft_filtered = plot_fft.plot(pen=pg.mkPen(COL_CYAN, width=1.5),
                                            name=f"FFT filtrat ({TIP_FILTRU})")

###############################################
########### Fereastra 2 - Sonometru octave ####
###############################################

# Etichete frecventa centru pentru bara
def format_fc(fc):
    if fc >= 1000:
        return f"{fc/1000:.2g}k"
    else:
        return f"{fc:.0f}"

win_octave = QtWidgets.QWidget()
win_octave.setWindowTitle(
    f"Sonometru - 1/{FRACTIE_OCTAVA} octava - {TIP_FILTRU}"
)
# Latimea depinde de numarul de benzi
bar_w = max(28, min(55, 1200 // (len(BENZI_OCTAVA) + 2)))
win_w = bar_w * (len(BENZI_OCTAVA) + 3) + 120
win_octave.resize(min(win_w, 1800), 620)
win_octave.setStyleSheet("background-color: black;")

layout_octave_main = QtWidgets.QHBoxLayout(win_octave)
layout_octave_main.setSpacing(2)

# Scala dB pe stanga
scala_widget = QtWidgets.QWidget()
scala_widget.setFixedWidth(45)
scala_layout = QtWidgets.QVBoxLayout(scala_widget)
scala_layout.setContentsMargins(0, 0, 0, 0)
scala_layout.setSpacing(0)
titlu_scala = QtWidgets.QLabel("dB FS")
titlu_scala.setStyleSheet("color: #aaa; font-size: 9px;")
titlu_scala.setAlignment(QtCore.Qt.AlignCenter)
scala_layout.addWidget(titlu_scala)
for val_db in [0, -10, -20, -30, -40, -60, -80, -100, -120]:
    lbl = QtWidgets.QLabel(f"{val_db}")
    lbl.setStyleSheet("color: #888; font-size: 8px;")
    lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    scala_layout.addWidget(lbl)
scala_layout.addStretch()
layout_octave_main.addWidget(scala_widget)

# Scroll area pentru benzile de octava (pot fi multe la 1/12)
scroll_area = QtWidgets.QScrollArea()
scroll_area.setWidgetResizable(True)
scroll_area.setStyleSheet("QScrollArea { border: none; background: black; }"
                          "QScrollBar:horizontal { height: 8px; }")
scroll_content = QtWidgets.QWidget()
scroll_content.setStyleSheet("background: black;")
layout_bare_octave = QtWidgets.QHBoxLayout(scroll_content)
layout_bare_octave.setSpacing(1)
layout_bare_octave.setContentsMargins(2, 2, 2, 2)

bare_octava = []
for fc, f_lo, f_hi in BENZI_OCTAVA:
    eticheta = format_fc(fc)
    col, bara, et_val, et_peak = creeaza_coloana_meter(eticheta, bar_w)
    layout_bare_octave.addLayout(col)
    bare_octava.append((bara, et_val, et_peak))

scroll_area.setWidget(scroll_content)
layout_octave_main.addWidget(scroll_area, stretch=5)

# Separator vertical
sep = QtWidgets.QFrame()
sep.setFrameShape(QtWidgets.QFrame.VLine)
sep.setStyleSheet("color: #444;")
layout_octave_main.addWidget(sep)

# Coloana dreapta: Nivel total dB al semnalului final (ponderat)
col_total, bara_total, et_total_val, et_total_peak = creeaza_coloana_meter(
    f"NIVEL\nTOTAL\n({TIP_FILTRU})", 55)
layout_octave_main.addLayout(col_total)

# Buton reset in win_octave
buton_reset_oct = QtWidgets.QPushButton("Reset Peak")
buton_reset_oct.setStyleSheet(
    "QPushButton { background-color: #333; color: white; padding: 5px; font-size: 10px; }"
    "QPushButton:hover { background-color: #555; }"
)
buton_reset_oct.clicked.connect(reseteaza_peak_hold)

layout_octave_v = QtWidgets.QVBoxLayout()
layout_octave_v.addWidget(win_octave)

win_octave.show()

# Pozitionam fereastra a doua langa prima
win_octave.move(50, 50)
win.move(win_octave.x(), win_octave.y() + win_octave.height() + 30)

###############################################
############## Date grafic ####################
###############################################

x_data = []
y_db_raw = []
y_db_filtered = []

stream = None
running = {"active": True}

def on_close_main(event=None):
    running["active"] = False

def on_close_oct(event=None):
    running["active"] = False

win.closeEvent = lambda event: (on_close_main(), event.accept())
win_octave.closeEvent = lambda event: (on_close_oct(), event.accept())

###############################################
############## Update GUI #####################
###############################################

def update_gui():
    global peak_hold_total

    if not running["active"]:
        return

    updated = False
    last_time = None
    last_db_raw = None
    last_db_filtered = None
    last_fft_raw = None
    last_fft_filtered = None
    last_niveluri_octava = None

    while True:
        try:
            packet = data_queue.get_nowait()
            (t, db_raw, db_filtered, fft_raw, fft_filtered, niveluri_octava) = packet
            last_time = t
            last_db_raw = db_raw
            last_db_filtered = db_filtered
            if not PEAK_MODE:
                x_data.append(t)
                y_db_raw.append(db_raw)
                y_db_filtered.append(db_filtered)
            last_fft_raw = fft_raw
            last_fft_filtered = fft_filtered
            last_niveluri_octava = niveluri_octava
            updated = True
        except queue.Empty:
            break

    if not updated:
        if stream is not None and not stream.active:
            running["active"] = False
            app.quit()
        return

    # Actualizeaza meter-ul principal
    peak_raw = get_peak_hold_display("raw", last_db_raw)
    peak_filt = get_peak_hold_display("filt", last_db_filtered)
    actualizeaza_bara(bara_raw, eticheta_raw, eticheta_peak_raw, last_db_raw, peak_raw)
    actualizeaza_bara(bara_filt, eticheta_filt, eticheta_peak_filt, last_db_filtered, peak_filt)

    # Actualizeaza sonometrul pe octave
    if last_niveluri_octava is not None:
        for i, db in enumerate(last_niveluri_octava):
            if db > peak_holds_octava[i]:
                peak_holds_octava[i] = db
            bara, et_val, et_peak = bare_octava[i]
            actualizeaza_bara(bara, et_val, et_peak, db, peak_holds_octava[i])

        # Nivel total = db_filtered (RMS al semnalului ponderat)
        if last_db_filtered > peak_hold_total:
            peak_hold_total = last_db_filtered
        actualizeaza_bara(bara_total, et_total_val, et_total_peak,
                          last_db_filtered, peak_hold_total)

    if PEAK_MODE:
        print(
            f"Timp: {last_time:6.2f}s | "
            f"Nefiltrat: {last_db_raw:6.1f} dBFS (peak {peak_raw:6.1f}) | "
            f"Filtrat ({TIP_FILTRU}): {last_db_filtered:6.1f} dBFS (peak {peak_filt:6.1f})"
        )
    else:
        if x_data:
            print(
                f"Timp: {x_data[-1]:6.2f}s | "
                f"Nefiltrat: {y_db_raw[-1]:6.1f} dBFS | "
                f"Filtrat: {y_db_filtered[-1]:6.1f} dBFS"
            )

        if plot_db is not None:
            if SURSA_OPT == "2" and x_data and x_data[-1] > 10:
                plot_db.setXRange(x_data[-1] - 10, x_data[-1], padding=0)
            curve_db_raw.setData(x_data, y_db_raw)
            curve_db_filtered.setData(x_data, y_db_filtered)

        if plot_fft is not None and last_fft_raw is not None:
            curve_fft_raw.setData(fft_frequencies_disp, last_fft_raw)
            curve_fft_filtered.setData(fft_frequencies_disp, last_fft_filtered)

    if stream is not None and not stream.active:
        running["active"] = False
        app.quit()

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(16)  # ~60 fps

###############################################
############# Pornire stream ##################
###############################################

try:
    if SURSA_OPT == "1":
        print(f"\nse reda audio ponderat {TIP_FILTRU} si se afiseaza live cu benzi 1/{FRACTIE_OCTAVA} octava")
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=playback_callback,
        )
    else:
        print(f"\nse analizeaza microfonul cu ponderare {TIP_FILTRU} si benzi 1/{FRACTIE_OCTAVA} octava")
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
    sd.stop()
    print("\nprocesare finalizata")

    if semnal_nefiltrat_complet and not PEAK_MODE:
        semnal_complet_raw = np.concatenate(semnal_nefiltrat_complet)
        semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

        N = len(semnal_complet_raw)
        freqs_finale = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)

        final_window = np.hanning(N)
        fft_orig = np.abs(np.fft.rfft(semnal_complet_raw * final_window)) / (N / 2.0)
        fft_filt = np.abs(np.fft.rfft(semnal_complet_filt * final_window)) / (N / 2.0)

        db_orig = np.clip(20 * np.log10(fft_orig + EPSILON), -120, 0)
        db_filt = np.clip(20 * np.log10(fft_filt + EPSILON), -120, 0)

        FINAL_DISPLAY_STEP = max(1, len(freqs_finale) // 4000)
        freqs_disp = freqs_finale[::FINAL_DISPLAY_STEP]
        db_orig_disp = db_orig[::FINAL_DISPLAY_STEP]
        db_filt_disp = db_filt[::FINAL_DISPLAY_STEP]

        titlu_final = f"Comparatie spectrala finala: original vs {TIP_FILTRU} (1/{FRACTIE_OCTAVA} oct)"

        app2 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        win_final = pg.GraphicsLayoutWidget(title=titlu_final)
        win_final.resize(1200, 550)
        win_final.show()

        plot_comp = win_final.addPlot(title=titlu_final)
        plot_comp.setLabel("bottom", "Frecventa (Hz)")
        plot_comp.setLabel("left", "Amplitudine (dB FS)")
        plot_comp.setLogMode(x=True, y=False)
        plot_comp.setYRange(-120, 0)
        plot_comp.showGrid(x=True, y=True, alpha=0.3)
        plot_comp.addLegend()

        max_pf = min(20000, nyquist)
        min_pf = min(20, max_pf / 10)
        plot_comp.setXRange(np.log10(min_pf), np.log10(max_pf))

        plot_comp.plot(freqs_disp, db_orig_disp,
                       pen=pg.mkPen(COL_ORANGE, width=1.5),
                       name="Spectru nefiltrat (original)")
        plot_comp.plot(freqs_disp, db_filt_disp,
                       pen=pg.mkPen(COL_CYAN, width=1.5),
                       name=f"Spectru ponderat ({TIP_FILTRU})")

        # Adauga linii verticale pentru frecventele centrale ale benzilor
        for fc, f_lo, f_hi in BENZI_OCTAVA:
            if min_pf < fc < max_pf:
                line = pg.InfiniteLine(
                    pos=np.log10(fc), angle=90,
                    pen=pg.mkPen((60, 60, 80), width=0.5, style=QtCore.Qt.DotLine),
                )
                plot_comp.addItem(line)

        app2.exec_()
    