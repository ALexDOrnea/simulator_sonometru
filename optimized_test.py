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

def proceseaza_ambele_semnale(chunk_raw, chunk_filtered):
    actualizeaza_ring_buffer(live_ring_buffer_raw, chunk_raw)
    actualizeaza_ring_buffer(live_ring_buffer_filtered, chunk_filtered)

    db_raw, fft_raw = calculeaza_db_fft(live_ring_buffer_raw)
    db_filtered, fft_filtered = calculeaza_db_fft(live_ring_buffer_filtered)

    return db_raw, db_filtered, fft_raw, fft_filtered

def trimite_date_live(chunk, chunk_filtrat):
    db_raw, db_filtered, fft_raw, fft_filtered = proceseaza_ambele_semnale(
        chunk, chunk_filtrat
    )
    current_time = play_pointer / SAMPLE_RATE
    # trimitem doar varianta decimata a fft-ului catre GUI (mai putin de desenat)
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

win = pg.GraphicsLayoutWidget(
    title=f"Analiza DSP live - {TIP_FILTRU}{titlu_frecv_str} - Mod: {MODE}"
)
win.resize(1100, 800)
win.show()

plot_db = None
plot_fft = None
curve_db_raw = curve_db_filtered = None
curve_fft_raw = curve_fft_filtered = None

# culori (RGBA) - orange / purple / cyan
COL_ORANGE = (255, 165, 0)
COL_PURPLE = (170, 90, 220)
COL_CYAN = (0, 220, 220)
COL_GRAY = (120, 120, 120)

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

def update_gui():
    if not running["active"]:
        return

    updated = False
    last_fft_raw = None
    last_fft_filtered = None

    # golim coada complet la fiecare tick de timer (nu doar un element)
    while True:
        try:
            (t, db_raw, db_filtered, fft_raw, fft_filtered) = data_queue.get_nowait()
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
    if semnal_nefiltrat_complet:
        import matplotlib.pyplot as plt

        semnal_complet_raw = np.concatenate(semnal_nefiltrat_complet)
        semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

        N = len(semnal_complet_raw)
        freqs_finale = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)

        final_window = np.hanning(N)
        fft_orig = np.abs(np.fft.rfft(semnal_complet_raw * final_window)) / (N / 2.0)
        fft_filt = np.abs(np.fft.rfft(semnal_complet_filt * final_window)) / (N / 2.0)

        db_orig = np.clip(20 * np.log10(fft_orig + EPSILON), -120, 0)
        db_filt = np.clip(20 * np.log10(fft_filt + EPSILON), -120, 0)

        fig2, ax_comp = plt.subplots(figsize=(10, 5))
        ax_comp.plot(
            freqs_finale,
            db_orig,
            color="orange",
            label="Spectru nefiltrat (original)",
        )
        ax_comp.plot(
            freqs_finale,
            db_filt,
            color="cyan",
            label=f"Spectru filtrat ({TIP_FILTRU})",
        )

        if CUTOFF_HZ is not None:
            ax_comp.axvline(
                CUTOFF_HZ,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"taiere ({CUTOFF_HZ:g} Hz)",
            )

        ax_comp.set_xscale("log")
        ax_comp.set_xlim(min(20, min(20000, nyquist) / 10), min(20000, nyquist))
        ax_comp.set_ylim(-120, 0)

        titlu_final = f"comparatie spectrala: original vs {TIP_FILTRU}"
        if CUTOFF_HZ is not None:
            titlu_final += f" @ {CUTOFF_HZ:g} Hz (ordin {ORDIN_FILTRU})"
        ax_comp.set_title(titlu_final)

        ax_comp.set_xlabel("Frecventa (Hz)")
        ax_comp.set_ylabel("Amplitudine (dB FS)")
        ax_comp.grid(True, which="both")
        ax_comp.legend()
        fig2.tight_layout()
        plt.show()