import warnings
import os
import sys
import time
import queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfilt_zi, butter, bilinear_zpk, zpk2sos

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
print("\n~~~~~SELECTIE PONDERARE (WEIGHTING)~~~~~")
print("1 A-Weighting (IEC 61672)")
print("2 C-Weighting (IEC 61672)")
print("3 Z-Weighting (flat, semnal nemodificat)")
PONDERARE_OPT = input("> ").strip().lower()

if PONDERARE_OPT in ("1", "a", "a-weighting", "a-weight"):
    TIP_PONDERARE = "A-Weighting"
elif PONDERARE_OPT in ("2", "c", "c-weighting", "c-weight"):
    TIP_PONDERARE = "C-Weighting"
elif PONDERARE_OPT in ("3", "z", "z-weighting", "z-weight"):
    TIP_PONDERARE = "Z-Weighting"
else:
    print("Optiune invalida. se foloseste A-Weighting")
    TIP_PONDERARE = "A-Weighting"

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

########################################################
############# FILTRE DE PONDERARE (IEC 61672) ##########
########################################################

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
    """Aplica ponderarea de frecventa selectata (A, C sau Z) semnalului brut."""
    global zi_ponderare
    if sos_ponderare is None:
        return chunk.copy()
    chunk_ponderat, zi_ponderare = sosfilt(sos_ponderare, chunk, zi=zi_ponderare)
    return chunk_ponderat

########################################################
##### BANC DE FILTRE PE BENZI DE OCTAVA (IEC 61260-1) ###
########################################################
# Seria de baza 10 (preferata de IEC 61260-1): raport de octava G = 10^(3/10)
# Frecventa centrala:  f_c = f_ref * G^(x/b)
# Limite de banda:     f_low = f_c * G^(-1/2b) ,  f_high = f_c * G^(1/2b)
# unde b = numarul de benzi pe octava (1, 3, 6 sau 12), f_ref = 1000 Hz

OCTAVE_G = 10 ** (3.0 / 10.0)
FREQ_REF = 1000.0

def formateaza_frecventa(f):
    if f >= 1000:
        return f"{f/1000:.3g}k"
    return f"{f:.3g}"

def design_octave_bands(fs, b, fmin=20.0, fmax=20000.0):
    """Proiecteaza un banc de filtre trece-banda Butterworth (sos) pe benzi de 1/b octava,
    conform frecventelor centrale/limitelor standardizate IEC 61260-1 (seria de baza 10)."""
    nyq = fs / 2.0
    fmax = min(fmax, nyq * 0.98)
    ordin = 4 if b <= 3 else 3  # ordin redus pentru benzi foarte inguste (stabilitate numerica)

    x_min = int(np.floor(b * np.log(fmin / FREQ_REF) / np.log(OCTAVE_G))) - 1
    x_max = int(np.ceil(b * np.log(fmax / FREQ_REF) / np.log(OCTAVE_G))) + 1

    benzi = []
    for x in range(x_min, x_max + 1):
        fc = FREQ_REF * OCTAVE_G ** (x / b)
        if fc < fmin * 0.95 or fc > fmax * 1.05:
            continue
        f_low = fc * OCTAVE_G ** (-1.0 / (2 * b))
        f_high = fc * OCTAVE_G ** (1.0 / (2 * b))
        if f_low < 1.0 or f_high >= nyq * 0.98:
            continue
        sos = butter(ordin, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        zi = sosfilt_zi(sos) * 0.0
        benzi.append({"x": x, "fc": fc, "f_low": f_low, "f_high": f_high, "sos": sos, "zi": zi})
    return benzi

octave_bands = design_octave_bands(SAMPLE_RATE, FRACTIE_OCTAVA)
NUM_BENZI = len(octave_bands)
band_ms_state = np.zeros(NUM_BENZI)

if NUM_BENZI == 0:
    print("Nu s-a putut genera nicio banda de frecventa pentru configuratia curenta.")
    sys.exit()

print(
    f"\nS-au generat {NUM_BENZI} benzi de 1/{FRACTIE_OCTAVA} octava "
    f"intre {octave_bands[0]['fc']:.1f} Hz si {octave_bands[-1]['fc']:.0f} Hz "
    f"(ponderare: {TIP_PONDERARE})"
)

def proceseaza_benzi(chunk_ponderat):
    """Filtreaza semnalul ponderat prin fiecare banda de octava (procesare in paralel/independenta
    pe fiecare banda) si calculeaza nivelul RMS ponderat in timp (Fast=0.125s, Slow=1s, Peak=0.035s),
    conform principiului de mediere exponentiala din IEC 61672-1."""
    global band_ms_state
    n = len(chunk_ponderat)
    if n == 0:
        return np.full(NUM_BENZI, -120.0)

    dt = n / SAMPLE_RATE
    if PEAK_MODE:
        tau = 0.035
    elif MODE.strip().lower() == "slow":
        tau = 1.0
    else:
        tau = 0.125
    alpha = 1.0 - np.exp(-dt / tau)

    niveluri = np.empty(NUM_BENZI)
    for i, banda in enumerate(octave_bands):
        filtrat, banda["zi"] = sosfilt(banda["sos"], chunk_ponderat, zi=banda["zi"])
        ms_bloc = np.mean(np.square(filtrat))
        band_ms_state[i] = (1.0 - alpha) * band_ms_state[i] + alpha * ms_bloc
        niveluri[i] = 10.0 * np.log10(band_ms_state[i] + EPSILON)
    return np.clip(niveluri, -120.0, 0.0)

########################################################
######### FERESTRE DE TIMP PENTRU METERUL GENERAL ######
########################################################

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

def proceseaza_ambele_semnale(chunk_raw, chunk_ponderat):
    actualizeaza_ring_buffer(live_ring_buffer_raw, chunk_raw)
    actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_ponderat)

    db_raw, fft_raw = calculeaza_db_fft(live_ring_buffer_raw)
    db_filtered, fft_filtered = calculeaza_db_fft(live_ring_buffer_filtered)

    return db_raw, db_filtered, fft_raw, fft_filtered

peak_hold_state = {
    "raw_db": -120.0,
    "filt_db": -120.0,
}

def get_peak_hold_display(cheie, valoare_curenta):
    val_key = f"{cheie}_db"
    if valoare_curenta > peak_hold_state[val_key]:
        peak_hold_state[val_key] = valoare_curenta
    return peak_hold_state[val_key]

def reseteaza_peak_hold():
    peak_hold_state["raw_db"] = -120.0
    peak_hold_state["filt_db"] = -120.0

def trimite_date_live(chunk, chunk_ponderat):
    current_time = play_pointer / SAMPLE_RATE

    if PEAK_MODE:
        actualizeaza_ring_buffer(live_ring_buffer_raw, chunk)
        actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_ponderat)
        db_raw = calculeaza_peak_db(live_ring_buffer_raw)
        db_filtered = calculeaza_peak_db(live_ring_buffer_filtered)
        data_queue.put((current_time, db_raw, db_filtered, None, None, None))
    else:
        db_raw, db_filtered, fft_raw, fft_filtered = proceseaza_ambele_semnale(
            chunk, chunk_ponderat
        )
        niveluri_benzi = proceseaza_benzi(chunk_ponderat) if plot_bands is not None else None
        data_queue.put((
            current_time,
            db_raw,
            db_filtered,
            fft_raw[::FFT_DISPLAY_STEP],
            fft_filtered[::FFT_DISPLAY_STEP],
            niveluri_benzi,
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

titlu_context_str = f" [{TIP_PONDERARE} | 1/{FRACTIE_OCTAVA} oct]"

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
    if db > -6.0:
        return "#e74c3c"
    elif db > -18.0:
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

    eticheta_valoare = QtWidgets.QLabel("-120.0 dB")
    eticheta_valoare.setStyleSheet("color: white; font-size: 12px;")
    eticheta_valoare.setAlignment(QtCore.Qt.AlignCenter)

    eticheta_peak = QtWidgets.QLabel("Peak: -120.0 dB")
    eticheta_peak.setStyleSheet("color: #f1c40f; font-size: 11px;")
    eticheta_peak.setAlignment(QtCore.Qt.AlignCenter)

    coloana.addWidget(eticheta_nume)
    coloana.addWidget(bara, alignment=QtCore.Qt.AlignHCenter)
    coloana.addWidget(eticheta_valoare)
    coloana.addWidget(eticheta_peak)
    return coloana, bara, eticheta_valoare, eticheta_peak

if PEAK_MODE:
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Live Peak Meter{titlu_context_str}")
    win.resize(360, 560)
    win.setStyleSheet("background-color: black;")

    layout_vertical = QtWidgets.QVBoxLayout(win)
    layout_principal = QtWidgets.QHBoxLayout()

    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw = creeaza_coloana_meter("Z (nefiltrat)")
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt = creeaza_coloana_meter(
        f"{TIP_PONDERARE}"
    )

    layout_principal.addLayout(coloana_raw)
    layout_principal.addLayout(coloana_filt)

    buton_reset = QtWidgets.QPushButton("Reset Peak")
    buton_reset.setStyleSheet(
        "QPushButton { background-color: #333; color: white; padding: 8px; font-size: 13px; }"
        "QPushButton:hover { background-color: #555; }"
        "QPushButton:pressed { background-color: #777; }"
    )
    buton_reset.clicked.connect(reseteaza_peak_hold)

    layout_vertical.addLayout(layout_principal)
    layout_vertical.addWidget(buton_reset)

    win.show()

else:
    # Fereastra hibrida: Grafice in stanga, Meter separat in dreapta
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Analiza DSP live{titlu_context_str} - Mod: {MODE}")
    win.resize(1280, 900)
    win.setStyleSheet("background-color: black;")

    layout_main = QtWidgets.QHBoxLayout(win)

    # Layout pyqtgraph pentru grafice
    win_graphics = pg.GraphicsLayoutWidget()
    layout_main.addWidget(win_graphics, stretch=4)

    # Panou separat pentru meter-ul de nivel
    panel_meter = QtWidgets.QWidget()
    layout_meter = QtWidgets.QVBoxLayout(panel_meter)

    layout_bars = QtWidgets.QHBoxLayout()
    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw = creeaza_coloana_meter("Z (nefiltrat)")
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt = creeaza_coloana_meter(
        f"{TIP_PONDERARE}"
    )

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

    afiseaza_db = GRAFIC_OPT in ("1", "4")
    afiseaza_fft = GRAFIC_OPT in ("2", "4")
    afiseaza_benzi = GRAFIC_OPT in ("3", "4")

    if afiseaza_db:
        plot_db = win_graphics.addPlot(title=f"Nivel live: Z (nefiltrat) vs {TIP_PONDERARE}")
        plot_db.setLabel("bottom", "Timp (s)")
        plot_db.setLabel("left", "Nivel (dB FS)")
        plot_db.setYRange(-120, 0)
        plot_db.showGrid(x=True, y=True, alpha=0.3)
        plot_db.addLegend()
        curve_db_raw = plot_db.plot(pen=pg.mkPen(COL_ORANGE, width=2), name=f"Z (nefiltrat) - nivel rolling ({MODE})")
        curve_db_filtered = plot_db.plot(pen=pg.mkPen(COL_PURPLE, width=2), name=f"{TIP_PONDERARE} - nivel rolling ({MODE})")

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
        plot_fft.setLabel("left", "Amplitudine (dB FS)")
        plot_fft.setLogMode(x=True, y=False)
        plot_fft.setYRange(-120, 0)
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
        plot_bands.setLabel("left", "Nivel (dB)")
        plot_bands.setYRange(-120, 0)
        plot_bands.showGrid(x=False, y=True, alpha=0.3)

        x_positions = np.arange(NUM_BENZI)
        bar_item = pg.BarGraphItem(
            x=x_positions, height=np.zeros(NUM_BENZI), width=0.8, brush=COL_GREEN, y0=-120.0
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

def actualizeaza_bara(bara, eticheta_valoare, eticheta_peak, valoare_db, valoare_peak_db):
    bara.setValue(int(np.clip((valoare_db + 120.0) * 10, 0, 1200)))
    bara.setStyleSheet(
        "QProgressBar { background-color: #222; border: 1px solid #555; }"
        f"QProgressBar::chunk {{ background-color: {culoare_pentru_nivel(valoare_db)}; }}"
    )
    eticheta_valoare.setText(f"{valoare_db:6.1f} dB")
    eticheta_peak.setText(f"Peak: {valoare_peak_db:6.1f} dB")

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

    while True:
        try:
            (t, db_raw, db_filtered, fft_raw, fft_filtered, niveluri_benzi) = data_queue.get_nowait()
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
    actualizeaza_bara(bara_raw, eticheta_raw, eticheta_peak_raw, last_db_raw, peak_raw)
    actualizeaza_bara(bara_filt, eticheta_filt, eticheta_peak_filt, last_db_filtered, peak_filt)

    if PEAK_MODE:
        print(
            f"Timp: {last_time:6.2f}s | "
            f"Z (nefiltrat): {last_db_raw:6.1f} dBFS (peak {peak_raw:6.1f}) | "
            f"{TIP_PONDERARE}: {last_db_filtered:6.1f} dBFS (peak {peak_filt:6.1f})"
        )
    else:
        print(f"Timp: {x_data[-1]:6.2f}s | Z (nefiltrat): {y_db_raw[-1]:6.1f} dBFS | {TIP_PONDERARE}: {y_db_filtered[-1]:6.1f} dBFS")

        if plot_db is not None:
            if SURSA_OPT == "2" and x_data[-1] > 10:
                plot_db.setXRange(x_data[-1] - 10, x_data[-1], padding=0)
            curve_db_raw.setData(x_data, y_db_raw)
            curve_db_filtered.setData(x_data, y_db_filtered)

        if plot_fft is not None and last_fft_raw is not None:
            curve_fft_raw.setData(fft_frequencies_disp, last_fft_raw)
            curve_fft_filtered.setData(fft_frequencies_disp, last_fft_filtered)

        if plot_bands is not None and last_niveluri_benzi is not None:
            bar_item.setOpts(height=last_niveluri_benzi + 120.0)

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

        titlu_final = f"comparatie spectrala: Z (nefiltrat) vs {TIP_PONDERARE}"

        app2 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        win2 = pg.GraphicsLayoutWidget(title=titlu_final)
        win2.resize(1100, 500)
        win2.show()

        plot_comp = win2.addPlot(title=titlu_final)
        plot_comp.setLabel("bottom", "Frecventa (Hz)")
        plot_comp.setLabel("left", "Amplitudine (dB FS)")
        plot_comp.setLogMode(x=True, y=False)
        plot_comp.setYRange(-120, 0)
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