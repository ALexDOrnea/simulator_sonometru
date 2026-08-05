import warnings
import os
import sys
import time
import queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, lfilter_zi, butter, sosfilt, sosfilt_zi, bilinear_zpk, zpk2sos

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

# NOU: flag global care spune daca suntem in modul Peak (live meter, fara grafice)
PEAK_MODE = MODE.strip().lower() == "peak"

print("\n~~~~~SELECTIE FILTRU~~~~~")
print("1 Low-pass")
print("2 High-pass")
print("3 A-Weighting (IEC 61672)")
print("4 C-Weighting (IEC 61672)")
FILTRU_OPT = input("> ").strip().lower()

CUTOFF_HZ = None
ORDIN_FILTRU = None
BTYPE_SCIPY = None

if FILTRU_OPT in ("1", "lowpass", "low-pass", "low", "lp"):
    TIP_FILTRU = "lowpass"
    BTYPE_SCIPY = "lowpass"
elif FILTRU_OPT in ("2", "highpass", "high-pass", "high", "hp"):
    TIP_FILTRU = "highpass"
    BTYPE_SCIPY = "highpass"
elif FILTRU_OPT in ("3", "a", "a-weighting", "a-weight"):
    TIP_FILTRU = "A-Weighting"
elif FILTRU_OPT in ("4", "c", "c-weighting", "c-weight"):
    TIP_FILTRU = "C-Weighting"
else:
    print("Invalid option. using highpass")
    TIP_FILTRU = "highpass"
    BTYPE_SCIPY = "highpass"

if TIP_FILTRU in ("highpass", "lowpass"):
    print("\n~~~~~SELECTIE FRECV TAIERE~~~~~")
    try:
        CUTOFF_HZ = float(input("Cutoff freq\n> ").strip())
    except ValueError:
        print("invalid freq")
        sys.exit()

    print("\n~~~~~SELECTIE ORDIN FILTRU~~~~~")
    try:
        ORDIN_FILTRU = float(input("filter order\n> ").strip())
    except ValueError:
        print("invalid value,using 4")
        ORDIN_FILTRU = 4

# MODIFICAT: in modul Peak nu mai are sens sa intrebam ce grafice vrea userul,
# pentru ca nu se mai afiseaza niciun grafic -- se afiseaza doar live meter-ul.
if PEAK_MODE:
    GRAFIC_OPT = None
    print("\n~~~~~MOD PEAK~~~~~")
    print("Se va afisa un live meter (fara grafice) cu maximul in dB, nefiltrat si filtrat.")
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

# Funcții pentru filtrele A și C
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

# Construim filtrul ales
if TIP_FILTRU in ("lowpass", "highpass"):
    if not 0 < CUTOFF_HZ < nyquist:
        print(f"cutoff freq must be between 0 and {nyquist:.1f} Hz.")
        sys.exit()
    cutoff_norm = CUTOFF_HZ / nyquist
    b_filter, a_filter = butter(ORDIN_FILTRU, cutoff_norm, btype=BTYPE_SCIPY)
    zi_filter = lfilter_zi(b_filter, a_filter) * 0.0
elif TIP_FILTRU == "A-Weighting":
    sos_filter = get_a_weighting_filter(SAMPLE_RATE)
    zi_filter = sosfilt_zi(sos_filter) * 0.0
elif TIP_FILTRU == "C-Weighting":
    sos_filter = get_c_weighting_filter(SAMPLE_RATE)
    zi_filter = sosfilt_zi(sos_filter) * 0.0

def filtreaza_block(chunk):
    """Filtrează un bloc și menține starea filtrului între apeluri"""
    global zi_filter
    if TIP_FILTRU in ("lowpass", "highpass"):
        chunk_filtrat, zi_filter = lfilter(b_filter, a_filter, chunk, zi=zi_filter)
    else:
        chunk_filtrat, zi_filter = sosfilt(sos_filter, chunk, zi=zi_filter)
    return chunk_filtrat

semnal_nefiltrat_complet = []
semnal_filtrat_complet = []

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

# decimare pentru afisarea FFT (nu afecteaza calculul, doar cate puncte se deseneaza)
FFT_DISPLAY_STEP = max(1, len(fft_frequencies) // 2000)
fft_frequencies_disp = fft_frequencies[::FFT_DISPLAY_STEP]

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

# NOU: pentru modul Peak folosim maximul absolut din fereastra, nu RMS-ul.
# Asta e diferenta reala dintre un "level meter" RMS si un adevarat peak meter.
def calculeaza_peak_db(buffer):
    peak = np.max(np.abs(buffer))
    return float(np.clip(20 * np.log10(peak + EPSILON), -120.0, 0.0))

def proceseaza_ambele_semnale(chunk_raw, chunk_filtered):
    actualizeaza_ring_buffer(live_ring_buffer_raw, chunk_raw)
    actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_filtered)

    db_raw, fft_raw = calculeaza_db_fft(live_ring_buffer_raw)
    db_filtered, fft_filtered = calculeaza_db_fft(live_ring_buffer_filtered)

    return db_raw, db_filtered, fft_raw, fft_filtered

# MODIFICAT: peak-hold "infinit" -- varful ramane fix la maximul atins pana la un reset
# manual sau pana cand apare o valoare si mai mare. Fara hold time, fara decay.
peak_hold_state = {
    "raw_db": -120.0,
    "filt_db": -120.0,
}

def get_peak_hold_display(cheie, valoare_curenta):
    """Actualizeaza (daca e cazul) si intoarce maximul retinut pentru acest canal."""
    val_key = f"{cheie}_db"
    if valoare_curenta > peak_hold_state[val_key]:
        peak_hold_state[val_key] = valoare_curenta
    return peak_hold_state[val_key]

def reseteaza_peak_hold():
    """Reseteaza ambele varfuri retinute la minim -- apelata de butonul Reset Peak."""
    peak_hold_state["raw_db"] = -120.0
    peak_hold_state["filt_db"] = -120.0

def trimite_date_live(chunk, chunk_filtrat):
    current_time = play_pointer / SAMPLE_RATE

    # MODIFICAT: in modul Peak nu mai calculam FFT (nu se mai afiseaza),
    # folosim direct maximul absolut din fereastra scurta.
    if PEAK_MODE:
        actualizeaza_ring_buffer(live_ring_buffer_raw, chunk)
        actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_filtrat)
        db_raw = calculeaza_peak_db(live_ring_buffer_raw)
        db_filtered = calculeaza_peak_db(live_ring_buffer_filtered)
        data_queue.put((current_time, db_raw, db_filtered, None, None))
    else:
        db_raw, db_filtered, fft_raw, fft_filtered = proceseaza_ambele_semnale(
            chunk, chunk_filtrat
        )
        data_queue.put((
            current_time,
            db_raw,
            db_filtered,
            fft_raw[::FFT_DISPLAY_STEP],
            fft_filtered[::FFT_DISPLAY_STEP],
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

titlu_frecv_str = f" @ {CUTOFF_HZ:g} Hz" if CUTOFF_HZ is not None else ""

plot_db = None
plot_fft = None
curve_db_raw = curve_db_filtered = None
curve_fft_raw = curve_fft_filtered = None

# culori (RGBA) - orange / purple / cyan
COL_ORANGE = (255, 165, 0)
COL_PURPLE = (170, 90, 220)
COL_CYAN = (0, 220, 220)
COL_GRAY = (120, 120, 120)

# NOU: praguri de culoare pentru live meter (verde/galben/rosu), ca la un VU-metru real
def culoare_pentru_nivel(db):
    if db > -6.0:
        return "#e74c3c"   # rosu -- foarte aproape de saturatie (0 dBFS)
    elif db > -18.0:
        return "#f1c40f"   # galben -- zona de atentie
    else:
        return "#2ecc71"   # verde -- nivel normal

bara_raw = bara_filt = None
eticheta_raw = eticheta_filt = None
eticheta_peak_raw = eticheta_peak_filt = None

if PEAK_MODE:
    # NOU: fereastra pentru modul Peak nu mai e un GraphicsLayoutWidget cu plot-uri,
    # ci un widget Qt simplu, cu doua bare verticale (VU-metre) -- una pt semnalul
    # nefiltrat, una pt cel filtrat -- plus marcaj numeric de varf (peak-hold).
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Live Peak Meter - {TIP_FILTRU}{titlu_frecv_str}")
    win.resize(360, 560)
    win.setStyleSheet("background-color: black;")

    # NOU: layout vertical care contine randul cu cele doua bare + butonul de reset dedesubt
    layout_vertical = QtWidgets.QVBoxLayout(win)
    layout_principal = QtWidgets.QHBoxLayout()

    def creeaza_coloana_meter(nume):
        coloana = QtWidgets.QVBoxLayout()

        eticheta_nume = QtWidgets.QLabel(nume)
        eticheta_nume.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        eticheta_nume.setAlignment(QtCore.Qt.AlignCenter)

        bara = QtWidgets.QProgressBar()
        bara.setOrientation(QtCore.Qt.Vertical)
        bara.setRange(0, 1200)   # reprezinta -120.0 .. 0.0 dB (o zecime de dB per unitate)
        bara.setValue(0)
        bara.setTextVisible(False)
        bara.setFixedWidth(70)
        bara.setStyleSheet(
            "QProgressBar { background-color: #222; border: 1px solid #555; }"
            "QProgressBar::chunk { background-color: #2ecc71; }"
        )

        eticheta_valoare = QtWidgets.QLabel("-120.0 dB")
        eticheta_valoare.setStyleSheet("color: white; font-size: 13px;")
        eticheta_valoare.setAlignment(QtCore.Qt.AlignCenter)

        eticheta_peak = QtWidgets.QLabel("Peak: -120.0 dB")
        eticheta_peak.setStyleSheet("color: #f1c40f; font-size: 12px;")
        eticheta_peak.setAlignment(QtCore.Qt.AlignCenter)

        coloana.addWidget(eticheta_nume)
        coloana.addWidget(bara, alignment=QtCore.Qt.AlignHCenter)
        coloana.addWidget(eticheta_valoare)
        coloana.addWidget(eticheta_peak)
        return coloana, bara, eticheta_valoare, eticheta_peak

    coloana_raw, bara_raw, eticheta_raw, eticheta_peak_raw = creeaza_coloana_meter("Nefiltrat")
    coloana_filt, bara_filt, eticheta_filt, eticheta_peak_filt = creeaza_coloana_meter(
        f"Filtrat ({TIP_FILTRU})"
    )

    layout_principal.addLayout(coloana_raw)
    layout_principal.addLayout(coloana_filt)

    # NOU: buton care reseteaza manual varfurile retinute (ambele canale deodata)
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
    win = pg.GraphicsLayoutWidget(
        title=f"Analiza DSP live - {TIP_FILTRU}{titlu_frecv_str} - Mod: {MODE}"
    )
    win.resize(1100, 800)
    win.show()

    if GRAFIC_OPT in ("1", "3"):
        plot_db = win.addPlot(title=f"Nivel live: nefiltrat vs {TIP_FILTRU}{titlu_frecv_str}")
        plot_db.setLabel("bottom", "Timp (s)")
        plot_db.setLabel("left", "Nivel (dB FS)")
        plot_db.setYRange(-120, 0)
        plot_db.showGrid(x=True, y=True, alpha=0.3)
        plot_db.addLegend()
        curve_db_raw = plot_db.plot(pen=pg.mkPen(COL_ORANGE, width=2), name=f"Nefiltrat - nivel rolling ({MODE})")
        curve_db_filtered = plot_db.plot(pen=pg.mkPen(COL_PURPLE, width=2), name=f"Filtrat ({TIP_FILTRU}) - nivel rolling ({MODE})")

        if SURSA_OPT == "1":
            duration = len(AUDIO_NORM) / SAMPLE_RATE
            plot_db.setXRange(0, duration)
        else:
            plot_db.setXRange(0, 10)

        if GRAFIC_OPT == "3":
            win.nextRow()

    if GRAFIC_OPT in ("2", "3"):
        plot_fft = win.addPlot(title=f"FFT in timp real: nefiltrat vs {TIP_FILTRU}{titlu_frecv_str}")
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
        curve_fft_filtered = plot_fft.plot(pen=pg.mkPen(COL_CYAN, width=1.5), name=f"FFT filtrat ({TIP_FILTRU})")

        if CUTOFF_HZ is not None:
            cutoff_line = pg.InfiniteLine(
                pos=np.log10(CUTOFF_HZ), angle=90,
                pen=pg.mkPen(COL_GRAY, width=1, style=QtCore.Qt.DashLine),
                label=f"Taiere: {CUTOFF_HZ:g} Hz",
            )
            plot_fft.addItem(cutoff_line)

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

# NOU: actualizeaza o bara-meter (culoare + valoare curenta + marcaj de varf)
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

    # golim coada complet la fiecare tick de timer (nu doar un element)
    while True:
        try:
            (t, db_raw, db_filtered, fft_raw, fft_filtered) = data_queue.get_nowait()
            last_time = t
            last_db_raw = db_raw
            last_db_filtered = db_filtered
            if not PEAK_MODE:
                x_data.append(t)
                y_db_raw.append(db_raw)
                y_db_filtered.append(db_filtered)
            last_fft_raw = fft_raw
            last_fft_filtered = fft_filtered
            updated = True
        except queue.Empty:
            break

    if not updated:
        if stream is not None and not stream.active:
            running["active"] = False
            app.quit()
        return

    # MODIFICAT: ramura separata pentru modul Peak -- actualizam meter-ul, nu grafice
    if PEAK_MODE:
        peak_raw = get_peak_hold_display("raw", last_db_raw)
        peak_filt = get_peak_hold_display("filt", last_db_filtered)

        actualizeaza_bara(bara_raw, eticheta_raw, eticheta_peak_raw, last_db_raw, peak_raw)
        actualizeaza_bara(bara_filt, eticheta_filt, eticheta_peak_filt, last_db_filtered, peak_filt)

        print(
            f"Timp: {last_time:6.2f}s | "
            f"Nefiltrat: {last_db_raw:6.1f} dBFS (peak {peak_raw:6.1f}) | "
            f"Filtrat: {last_db_filtered:6.1f} dBFS (peak {peak_filt:6.1f})"
        )
    else:
        print(f"Timp: {x_data[-1]:6.2f}s | Nefiltrat: {y_db_raw[-1]:6.1f} dBFS | Filtrat: {y_db_filtered[-1]:6.1f} dBFS")

        if plot_db is not None:
            if SURSA_OPT == "2" and x_data[-1] > 10:
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

try:
    if SURSA_OPT == "1":
        print(f"se reda audio filtrat {TIP_FILTRU}{titlu_frecv_str} si se afiseaza comparatia live")
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=playback_callback,
        )
    else:
        print(f"se analizeaza microfonul cu filtru {TIP_FILTRU}{titlu_frecv_str} si se afiseaza comparatia live")
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

    ########################################################
    ######## Comparatie spectrala pe tot semnalul #########
    # MODIFICAT: in modul Peak nu mai deschidem nicio fereastra cu grafice la final,
    # ramanem consecventi cu ideea de "fara grafice" ceruta pentru acest mod.
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

        # decimam si graficul final, la fel ca la live, ca sa deseneze rapid
        FINAL_DISPLAY_STEP = max(1, len(freqs_finale) // 4000)
        freqs_disp = freqs_finale[::FINAL_DISPLAY_STEP]
        db_orig_disp = db_orig[::FINAL_DISPLAY_STEP]
        db_filt_disp = db_filt[::FINAL_DISPLAY_STEP]

        titlu_final = f"comparatie spectrala: original vs {TIP_FILTRU}"
        if CUTOFF_HZ is not None:
            titlu_final += f" @ {CUTOFF_HZ:g} Hz (ordin {ORDIN_FILTRU})"

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
            name="Spectru nefiltrat (original)",
        )
        plot_comp.plot(
            freqs_disp, db_filt_disp,
            pen=pg.mkPen(COL_CYAN, width=1.5),
            name=f"Spectru filtrat ({TIP_FILTRU})",
        )

        if CUTOFF_HZ is not None:
            cutoff_line_final = pg.InfiniteLine(
                pos=np.log10(CUTOFF_HZ), angle=90,
                pen=pg.mkPen(COL_GRAY, width=1, style=QtCore.Qt.DashLine),
                label=f"taiere ({CUTOFF_HZ:g} Hz)",
            )
            plot_comp.addItem(cutoff_line_final)

        app2.exec_()