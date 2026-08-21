# ============================================================
# SONOMETRU REALTIME
# ============================================================
#
# Functionalitati:
#   - input Live sau WAV
#   - Fast / Slow / Peak
#   - A / C / Z weighting
#   - 1/1, 1/3, 1/6, 1/12 octave
#   - calibrare SPL cu offset in dB
#   - corectie de raspuns in frecventa a microfonului
#   - FFT live
#   - benzi de octava live
#   - Leq
#   - Peak hold
#
# Corectia microfonului NU este un offset global.
# Ea este aplicata dependent de frecventa:
#
#     nivel_corectat = nivel_masurat - raspuns_microfon_dB
#
# Fisierul implicit este:
#     35Y228_cal_0degree(1).txt
# ============================================================

import os
import sys
import time
import queue
import threading
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import (
    sosfilt,
    sosfilt_zi,
    butter,
    bilinear_zpk,
    zpk2sos,
    lfilter,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="scipy.io.wavfile",
)

try:
    import sounddevice as sd
except ImportError:
    print("Lipseste sounddevice. Instaleaza: pip install sounddevice")
    sys.exit(1)

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("Lipseste pyqtgraph/PyQt5. Instaleaza: pip install pyqtgraph pyqt5")
    sys.exit(1)

try:
    from microphone_calibration import MicrophoneCalibration
except ImportError:
    print(
        "Nu gasesc microphone_calibration.py. "
        "Pune fisierul in acelasi folder cu main.py."
    )
    sys.exit(1)


# ============================================================
# CONFIGURARE
# ============================================================

SAMPLE_RATE_DEFAULT = 48000
BLOCKSIZE = 256
LATENCY = "low"

GUI_REFRESH_MS = 25
BENZI_FFT_REFRESH_HZ = 20.0

EPSILON = 1e-12

DEFAULT_CALIBRATION_FILE = "35Y228_cal_0degree(1).txt"

# Corectia este activata implicit.
MIC_CALIBRATION_ENABLED = True

# Domeniul util de afisare.
LEVEL_MIN = -120.0
LEVEL_MAX = 20.0

# ============================================================
# SELECTARE DEVICE
# ============================================================


def select_audio_devices():
    devices = sd.query_devices()

    input_devices = []
    output_devices = []

    print("\n================ CONFIG AUDIO ================\n")
    print("INPUT:")

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(
                f"[{i}] {dev['name']} "
                f"(input channels: {dev['max_input_channels']})"
            )
            input_devices.append(i)

    print("\nOUTPUT:")

    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            print(
                f"[{i}] {dev['name']} "
                f"(output channels: {dev['max_output_channels']})"
            )
            output_devices.append(i)

    input_id = None
    output_id = None

    try:
        value = input(
            "\nID input (Enter = default): "
        ).strip()

        if value:
            candidate = int(value)
            if candidate in input_devices:
                input_id = candidate
            else:
                print("ID input invalid. Se foloseste default.")

        value = input(
            "ID output (Enter = default): "
        ).strip()

        if value:
            candidate = int(value)
            if candidate in output_devices:
                output_id = candidate
            else:
                print("ID output invalid. Se foloseste default.")

    except ValueError:
        print("Valoare invalida. Se folosesc device-urile default.")

    sd.default.device = (input_id, output_id)

    return input_id, output_id


# ============================================================
# CONFIGURARE UTILIZATOR
# ============================================================


def ask_configuration():
    print("\n================ SURSA SEMNAL ================\n")
    print("1. WAV")
    print("2. Live")

    source = input("> ").strip()

    wav_path = None

    if source == "1":
        wav_path = input("Path WAV: ").strip().strip('"')

        if not os.path.exists(wav_path):
            print(f"Fisierul nu exista: {wav_path}")
            sys.exit(1)

    elif source != "2":
        print("Optiune invalida.")
        sys.exit(1)

    print("\n================ MOD ================\n")
    mode = input("Mode (Fast, Slow, Peak): ").strip().lower()

    if mode not in ("fast", "slow", "peak"):
        print("Mode invalid. Se foloseste Fast.")
        mode = "fast"

    print("\n================ WEIGHTING ================\n")
    print("1. A-Weighting")
    print("2. C-Weighting")
    print("3. Z-Weighting")

    weighting = input("> ").strip().lower()

    if weighting in ("1", "a", "a-weighting", "a-weight"):
        weighting_type = "A-Weighting"
        weighting_symbol = "A"
    elif weighting in ("2", "c", "c-weighting", "c-weight"):
        weighting_type = "C-Weighting"
        weighting_symbol = "C"
    elif weighting in ("3", "z", "z-weighting", "z-weight"):
        weighting_type = "Z-Weighting"
        weighting_symbol = "Z"
    else:
        print("Optiune invalida. Se foloseste A-Weighting.")
        weighting_type = "A-Weighting"
        weighting_symbol = "A"

    print("\n================ BENZI ================\n")
    print("1. Octave 1/1")
    print("2. Terte de octava 1/3")
    print("3. Sesimi de octava 1/6")
    print("4. Doisprezecimi 1/12")

    band_option = input("> ").strip()
    band_fraction = {
        "1": 1,
        "2": 3,
        "3": 6,
        "4": 12,
    }.get(band_option, 3)

    if band_option not in ("1", "2", "3", "4"):
        print("Optiune invalida. Se foloseste 1/3 octava.")

    print("\n================ GRAFICE ================\n")

    if mode == "peak":
        graph_option = "1"
        print("Peak: meter live.")
    else:
        print("1. Nivel in timp")
        print("2. FFT")
        print("3. Benzi de octava")
        print("4. Toate")

        graph_option = input("> ").strip()

        if graph_option not in ("1", "2", "3", "4"):
            print("Optiune invalida. Se afiseaza toate.")
            graph_option = "4"

    print("\n================ CALIBRARE SPL ================\n")
    print("Offsetul SPL se determina cu un calibrator acustic.")
    print("Exemplu: 94 dB SPL la 1 kHz.")
    print("CALIBRARE_DB = SPL_cunoscut - dBFS_masurat.")

    try:
        calibration_db = float(
            input("Constanta calibrare SPL [dB]: ")
            .strip()
            .replace(",", ".")
        )
    except ValueError:
        calibration_db = 0.0
        print("Valoare invalida. Se foloseste 0 dB.")

    print(
        f"Calibrare SPL folosita: "
        f"{calibration_db:+.2f} dB"
    )

    return {
        "source": source,
        "wav_path": wav_path,
        "mode": mode,
        "weighting_type": weighting_type,
        "weighting_symbol": weighting_symbol,
        "band_fraction": band_fraction,
        "graph_option": graph_option,
        "calibration_db": calibration_db,
    }


# ============================================================
# CITIRE WAV
# ============================================================


def normalize_audio(audio):
    if audio.ndim > 1:
        audio = audio[:, 0]

    if audio.dtype == np.int16:
        return audio.astype(np.float64) / 32768.0

    if audio.dtype == np.int32:
        return audio.astype(np.float64) / 2147483648.0

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return audio.astype(np.float64) / scale

    return audio.astype(np.float64)


# ============================================================
# A / C WEIGHTING
# ============================================================


def prewarp_frequency(frequency, fs):
    return (
        fs
        / np.pi
        * np.tan(np.pi * frequency / fs)
    )


def get_a_weighting_filter(fs):
    f1, f2, f3, f4 = (
        20.598997,
        107.65265,
        737.86223,
        12194.217,
    )

    f1, f2, f3, f4 = [
        prewarp_frequency(f, fs)
        for f in (f1, f2, f3, f4)
    ]

    A1000 = 1.9997

    p1 = -2 * np.pi * f1
    p2 = -2 * np.pi * f2
    p3 = -2 * np.pi * f3
    p4 = -2 * np.pi * f4

    zeros = [0, 0, 0, 0]
    poles = [p1, p1, p2, p3, p4, p4]

    gain = (
        (2 * np.pi * f4) ** 2
        * 10 ** (A1000 / 20)
    )

    zd, pd, kd = bilinear_zpk(
        zeros,
        poles,
        gain,
        fs,
    )

    return zpk2sos(zd, pd, kd)


def get_c_weighting_filter(fs):
    f1, f4 = 20.598997, 12194.217

    f1 = prewarp_frequency(f1, fs)
    f4 = prewarp_frequency(f4, fs)

    C1000 = 0.0619

    p1 = -2 * np.pi * f1
    p4 = -2 * np.pi * f4

    zeros = [0, 0]
    poles = [p1, p1, p4, p4]

    gain = (
        (2 * np.pi * f4) ** 2
        * 10 ** (C1000 / 20)
    )

    zd, pd, kd = bilinear_zpk(
        zeros,
        poles,
        gain,
        fs,
    )

    return zpk2sos(zd, pd, kd)


# ============================================================
# FILTRARE
# ============================================================


def create_weighting_filter(weighting_type, fs):
    if weighting_type == "A-Weighting":
        return get_a_weighting_filter(fs)

    if weighting_type == "C-Weighting":
        return get_c_weighting_filter(fs)

    return None


# ============================================================
# TIME WEIGHTING
# ============================================================


def create_time_filter(tau, fs):
    alpha = 1.0 - np.exp(
        -1.0 / (tau * fs)
    )

    b = [alpha]
    a = [1.0, -(1.0 - alpha)]

    return b, a


# ============================================================
# OCTAVE BANDS
# ============================================================

OCTAVE_G = 10 ** (3.0 / 10.0)
FREQ_REF = 1000.0


def exact_center_frequency(x, bands_per_octave):
    if bands_per_octave % 2 == 1:
        return (
            FREQ_REF
            * OCTAVE_G ** (x / bands_per_octave)
        )

    return (
        FREQ_REF
        * OCTAVE_G
        ** ((2 * x + 1) / (2 * bands_per_octave))
    )


def design_octave_bands(
    fs,
    bands_per_octave,
    fmin=20.0,
    fmax=20000.0,
):
    nyquist = fs / 2.0
    fmax = min(fmax, nyquist * 0.98)

    order = 12

    x_min = int(
        np.floor(
            bands_per_octave
            * np.log(fmin / FREQ_REF)
            / np.log(OCTAVE_G)
        )
    ) - 1

    x_max = int(
        np.ceil(
            bands_per_octave
            * np.log(fmax / FREQ_REF)
            / np.log(OCTAVE_G)
        )
    ) + 1

    bands = []

    for x in range(x_min, x_max + 1):
        fc = exact_center_frequency(
            x,
            bands_per_octave,
        )

        if (
            fc < fmin * 0.95
            or fc > fmax * 1.05
        ):
            continue

        f_low = (
            fc
            * OCTAVE_G
            ** (-1.0 / (2 * bands_per_octave))
        )

        f_high = (
            fc
            * OCTAVE_G
            ** (1.0 / (2 * bands_per_octave))
        )

        if (
            f_low < 1.0
            or f_high >= nyquist * 0.98
        ):
            continue

        sos = butter(
            order,
            [f_low, f_high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        bands.append(
            {
                "fc": fc,
                "f_low": f_low,
                "f_high": f_high,
                "sos": sos,
                "zi_bandpass": (
                    sosfilt_zi(sos) * 0.0
                ),
            }
        )

    return bands


# ============================================================
# METER ENGINE
# ============================================================


class MeterEngine:
    def __init__(
        self,
        fs,
        mode,
        weighting_type,
        band_fraction,
        calibration_db,
        microphone_calibration,
    ):
        self.fs = fs
        self.mode = mode
        self.weighting_type = weighting_type
        self.band_fraction = band_fraction
        self.calibration_db = calibration_db
        self.mic_calibration = microphone_calibration

        self.nyquist = fs / 2.0

        self.weighting_sos = create_weighting_filter(
            weighting_type,
            fs,
        )

        self.weighting_zi = (
            sosfilt_zi(self.weighting_sos) * 0.0
            if self.weighting_sos is not None
            else None
        )

        tau = 1.0 if mode == "slow" else 0.125

        self.b_time, self.a_time = (
            create_time_filter(tau, fs)
        )

        self.zi_raw = [0.0]
        self.zi_weighted = [0.0]

        self.bands = design_octave_bands(
            fs,
            band_fraction,
        )

        self.band_b_time = []
        self.band_a_time = []
        self.band_zi_time = []

        for _ in self.bands:
            b, a = create_time_filter(
                tau,
                fs,
            )

            self.band_b_time.append(b)
            self.band_a_time.append(a)
            self.band_zi_time.append([0.0])

        self.leq_sum_raw = 0.0
        self.leq_sum_weighted = 0.0
        self.leq_samples = 0

        self.peak_raw = LEVEL_MIN
        self.peak_weighted = LEVEL_MIN

    def reset(self):
        self.leq_sum_raw = 0.0
        self.leq_sum_weighted = 0.0
        self.leq_samples = 0

        self.peak_raw = LEVEL_MIN
        self.peak_weighted = LEVEL_MIN

    def weighting_filter(self, chunk):
        if self.weighting_sos is None:
            return chunk.copy()

        filtered, self.weighting_zi = sosfilt(
            self.weighting_sos,
            chunk,
            zi=self.weighting_zi,
        )

        return filtered

    def db_from_ms(self, mean_square):
        value = (
            10.0
            * np.log10(mean_square + EPSILON)
            + self.calibration_db
        )

        return float(
            np.clip(
                value,
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )

    def update_leq(self, raw, weighted):
        self.leq_sum_raw += float(
            np.sum(raw * raw)
        )

        self.leq_sum_weighted += float(
            np.sum(weighted * weighted)
        )

        self.leq_samples += len(raw)

    def calculate_leq(self):
        if self.leq_samples == 0:
            return LEVEL_MIN, LEVEL_MIN

        raw_ms = (
            self.leq_sum_raw
            / self.leq_samples
        )

        weighted_ms = (
            self.leq_sum_weighted
            / self.leq_samples
        )

        raw_db = self.db_from_ms(raw_ms)
        weighted_db = self.db_from_ms(
            weighted_ms
        )

        return raw_db, weighted_db

    def process_bands(self, weighted_chunk):
        levels = np.full(
            len(self.bands),
            LEVEL_MIN,
            dtype=np.float64,
        )

        for i, band in enumerate(self.bands):
            filtered, band["zi_bandpass"] = (
                sosfilt(
                    band["sos"],
                    weighted_chunk,
                    zi=band["zi_bandpass"],
                )
            )

            ms, self.band_zi_time[i] = lfilter(
                self.band_b_time[i],
                self.band_a_time[i],
                filtered * filtered,
                zi=self.band_zi_time[i],
            )

            measured_db = self.db_from_ms(
                ms[-1]
            )

            corrected_db = self.mic_calibration.correct_db(
                measured_db,
                band["fc"],
            )

            levels[i] = float(
                np.clip(
                    corrected_db,
                    LEVEL_MIN,
                    LEVEL_MAX,
                )
            )

        return levels

    def process_level(self, raw, weighted):
        self.update_leq(raw, weighted)

        leq_raw, leq_weighted = (
            self.calculate_leq()
        )

        if self.mode == "peak":
            raw_peak = float(
                np.max(np.abs(raw))
            )

            weighted_peak = float(
                np.max(np.abs(weighted))
            )

            raw_db = (
                20
                * np.log10(
                    raw_peak + EPSILON
                )
                + self.calibration_db
            )

            weighted_db = (
                20
                * np.log10(
                    weighted_peak + EPSILON
                )
                + self.calibration_db
            )

            raw_db = float(
                np.clip(
                    raw_db,
                    LEVEL_MIN,
                    LEVEL_MAX,
                )
            )

            weighted_db = float(
                np.clip(
                    weighted_db,
                    LEVEL_MIN,
                    LEVEL_MAX,
                )
            )

            self.peak_raw = max(
                self.peak_raw,
                raw_db,
            )

            self.peak_weighted = max(
                self.peak_weighted,
                weighted_db,
            )

            return (
                raw_db,
                weighted_db,
                leq_raw,
                leq_weighted,
            )

        raw_ms, self.zi_raw = lfilter(
            self.b_time,
            self.a_time,
            raw * raw,
            zi=self.zi_raw,
        )

        weighted_ms, self.zi_weighted = (
            lfilter(
                self.b_time,
                self.a_time,
                weighted * weighted,
                zi=self.zi_weighted,
            )
        )

        raw_db = self.db_from_ms(
            raw_ms[-1]
        )

        weighted_db = self.db_from_ms(
            weighted_ms[-1]
        )

        return (
            raw_db,
            weighted_db,
            leq_raw,
            leq_weighted,
        )


# ============================================================
# FFT
# ============================================================


def calculate_fft(
    signal,
    fs,
    calibration_db,
    mic_calibration,
):
    n = len(signal)

    if n < 2:
        return (
            np.array([]),
            np.array([]),
        )

    window = np.hanning(n)

    spectrum = np.abs(
        np.fft.rfft(
            signal * window
        )
    )

    spectrum /= max(
        n / 2.0,
        1.0,
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    db = (
        20
        * np.log10(
            spectrum + EPSILON
        )
        + calibration_db
    )

    # Corectia de frecventa se aplica pe fiecare
    # bin FFT. Nu modificam semnalul audio.
    db = mic_calibration.correct_fft_db(
        db,
        frequencies,
    )

    db = np.clip(
        db,
        LEVEL_MIN,
        LEVEL_MAX,
    )

    return frequencies, db


# ============================================================
# AUDIO PROCESSING THREADS
# ============================================================


class RealtimeProcessor:
    def __init__(
        self,
        engine,
        source,
        audio_data=None,
    ):
        self.engine = engine
        self.source = source
        self.audio_data = audio_data

        self.raw_queue = queue.Queue(
            maxsize=4
        )

        self.data_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.sample_pointer = 0

        self.batch_raw = []
        self.batch_weighted = []
        self.batch_count = 0

        self.fft_window_size = int(
            0.125 * engine.fs
        )

        if engine.mode == "slow":
            self.fft_window_size = int(
                1.0 * engine.fs
            )
        elif engine.mode == "peak":
            self.fft_window_size = int(
                0.035 * engine.fs
            )

        self.fft_window_size = max(
            self.fft_window_size,
            1024,
        )

        self.fft_raw_buffer = np.zeros(
            self.fft_window_size,
            dtype=np.float64,
        )

        self.fft_weighted_buffer = (
            np.zeros(
                self.fft_window_size,
                dtype=np.float64,
            )
        )

        self.batch_factor = max(
            1,
            round(
                (
                    1.0
                    / BENZI_FFT_REFRESH_HZ
                )
                * engine.fs
                / BLOCKSIZE
            ),
        )

    def update_ring_buffer(
        self,
        buffer,
        chunk,
    ):
        frames = len(chunk)

        if frames >= len(buffer):
            buffer[:] = chunk[
                -len(buffer):
            ]
            return

        buffer[:-frames] = buffer[
            frames:
        ]

        buffer[-frames:] = chunk

    def calculate_display_data(
        self,
        raw_chunk,
        weighted_chunk,
    ):
        fft_raw = None
        fft_weighted = None
        band_levels = None

        self.batch_raw.append(
            raw_chunk
        )

        self.batch_weighted.append(
            weighted_chunk
        )

        self.batch_count += 1

        if self.batch_count < self.batch_factor:
            return (
                fft_raw,
                fft_weighted,
                band_levels,
            )

        raw = np.concatenate(
            self.batch_raw
        )

        weighted = np.concatenate(
            self.batch_weighted
        )

        self.batch_raw.clear()
        self.batch_weighted.clear()
        self.batch_count = 0

        self.update_ring_buffer(
            self.fft_raw_buffer,
            raw,
        )

        self.update_ring_buffer(
            self.fft_weighted_buffer,
            weighted,
        )

        fft_frequencies, fft_raw = (
            calculate_fft(
                self.fft_raw_buffer,
                self.engine.fs,
                self.engine.calibration_db,
                self.engine.mic_calibration,
            )
        )

        _, fft_weighted = calculate_fft(
            self.fft_weighted_buffer,
            self.engine.fs,
            self.engine.calibration_db,
            self.engine.mic_calibration,
        )

        band_levels = (
            self.engine.process_bands(
                weighted
            )
        )

        return (
            fft_frequencies,
            fft_raw,
            fft_weighted,
            band_levels,
        )

    def processing_loop(self):
        while True:
            try:
                (
                    raw,
                    weighted,
                    pointer,
                ) = self.raw_queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            (
                db_raw,
                db_weighted,
                leq_raw,
                leq_weighted,
            ) = self.engine.process_level(
                raw,
                weighted,
            )

            display = (
                self.calculate_display_data(
                    raw,
                    weighted,
                )
            )

            if len(display) == 3:
                fft_freq = None
                fft_raw = None
                fft_weighted = None
                band_levels = None
            else:
                (
                    fft_freq,
                    fft_raw,
                    fft_weighted,
                    band_levels,
                ) = display

            self.data_queue.put(
                (
                    pointer / self.engine.fs,
                    db_raw,
                    db_weighted,
                    fft_freq,
                    fft_raw,
                    fft_weighted,
                    band_levels,
                    leq_raw,
                    leq_weighted,
                )
            )


# ============================================================
# GUI
# ============================================================


class MeterWindow(QtWidgets.QWidget):
    def __init__(
        self,
        engine,
        processor,
        graph_option,
    ):
        super().__init__()

        self.engine = engine
        self.processor = processor
        self.graph_option = graph_option

        self.time_x = []
        self.time_raw = []
        self.time_weighted = []

        self.max_time_points = 600

        self.band_bars = None
        self.band_x = None

        self.last_fft_freq = None
        self.last_fft_raw = None
        self.last_fft_weighted = None

        self.running = True

        self.setWindowTitle(
            self.make_title()
        )

        self.resize(1400, 900)

        self.build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(
            self.update_gui
        )
        self.timer.start(
            GUI_REFRESH_MS
        )

    def make_title(self):
        mic_text = (
            self.engine.mic_calibration.description()
        )

        return (
            "Sonometrul realtime | "
            f"{self.engine.weighting_type} | "
            f"1/{self.engine.band_fraction} oct | "
            f"{mic_text}"
        )

    def build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(
            self
        )

        graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(
            graphics,
            stretch=5,
        )

        meter_panel = QtWidgets.QWidget()
        meter_layout = QtWidgets.QVBoxLayout(
            meter_panel
        )

        main_layout.addWidget(
            meter_panel,
            stretch=1,
        )

        self.value_raw = QtWidgets.QLabel(
            "Z: --.- dB"
        )

        self.value_weighted = (
            QtWidgets.QLabel(
                f"{self.engine.weighting_type}: --.- dB"
            )
        )

        self.peak_raw_label = (
            QtWidgets.QLabel(
                "Peak Z: --.- dB"
            )
        )

        self.peak_weighted_label = (
            QtWidgets.QLabel(
                "Peak weighted: --.- dB"
            )
        )

        self.leq_raw_label = (
            QtWidgets.QLabel(
                "Leq Z: --.- dB"
            )
        )

        self.leq_weighted_label = (
            QtWidgets.QLabel(
                "Leq weighted: --.- dB"
            )
        )

        for label in (
            self.value_raw,
            self.value_weighted,
            self.peak_raw_label,
            self.peak_weighted_label,
            self.leq_raw_label,
            self.leq_weighted_label,
        ):
            label.setStyleSheet(
                "color: white; "
                "font-size: 16px; "
                "padding: 5px;"
            )
            meter_layout.addWidget(
                label
            )

        self.mic_label = QtWidgets.QLabel(
            self.engine.mic_calibration.description()
        )

        self.mic_label.setStyleSheet(
            "color: #00dddd; "
            "font-size: 14px; "
            "padding: 8px;"
        )

        meter_layout.addWidget(
            self.mic_label
        )

        reset_button = QtWidgets.QPushButton(
            "Reset Peak / Leq"
        )

        reset_button.clicked.connect(
            self.reset_meter
        )

        meter_layout.addWidget(
            reset_button
        )

        if self.graph_option in (
            "1",
            "4",
        ):
            self.plot_time = (
                graphics.addPlot(
                    title="Nivel in timp"
                )
            )

            self.plot_time.setLabel(
                "bottom",
                "Timp",
                units="s",
            )

            self.plot_time.setLabel(
                "left",
                "Nivel",
                units="dB",
            )

            self.plot_time.setYRange(
                LEVEL_MIN,
                LEVEL_MAX,
            )

            self.plot_time.showGrid(
                x=True,
                y=True,
                alpha=0.3,
            )

            self.curve_time_raw = (
                self.plot_time.plot(
                    pen=pg.mkPen(
                        (255, 165, 0),
                        width=2,
                    ),
                    name="Z",
                )
            )

            self.curve_time_weighted = (
                self.plot_time.plot(
                    pen=pg.mkPen(
                        (0, 220, 220),
                        width=2,
                    ),
                    name=self.engine.weighting_type,
                )
            )

            self.plot_time.addLegend()

        else:
            self.plot_time = None

        if self.graph_option in (
            "2",
            "4",
        ):
            self.plot_fft = (
                graphics.addPlot(
                    title="FFT - raspuns corectat"
                )
            )

            self.plot_fft.setLabel(
                "bottom",
                "Frecventa",
                units="Hz",
            )

            self.plot_fft.setLabel(
                "left",
                "Nivel",
                units="dB",
            )

            self.plot_fft.setLogMode(
                x=True,
                y=False,
            )

            self.plot_fft.setYRange(
                LEVEL_MIN,
                LEVEL_MAX,
            )

            self.plot_fft.showGrid(
                x=True,
                y=True,
                alpha=0.3,
            )

            self.curve_fft_raw = (
                self.plot_fft.plot(
                    pen=pg.mkPen(
                        (255, 165, 0),
                        width=1.5,
                    ),
                    name="Z corectat",
                )
            )

            self.curve_fft_weighted = (
                self.plot_fft.plot(
                    pen=pg.mkPen(
                        (0, 220, 220),
                        width=1.5,
                    ),
                    name="Weighted corectat",
                )
            )

            self.plot_fft.addLegend()

        else:
            self.plot_fft = None

        if self.graph_option in (
            "3",
            "4",
        ):
            self.plot_bands = (
                graphics.addPlot(
                    title="Benzi de octava - corectate"
                )
            )

            self.plot_bands.setLabel(
                "bottom",
                "Frecventa centrala",
                units="Hz",
            )

            self.plot_bands.setLabel(
                "left",
                "Nivel",
                units="dB",
            )

            self.plot_bands.setLogMode(
                x=True,
                y=False,
            )

            self.plot_bands.setYRange(
                LEVEL_MIN,
                LEVEL_MAX,
            )

            self.plot_bands.showGrid(
                x=True,
                y=True,
                alpha=0.3,
            )

            self.band_curve = (
                self.plot_bands.plot(
                    pen=pg.mkPen(
                        (170, 90, 220),
                        width=2,
                    ),
                    symbol="o",
                    symbolSize=5,
                )
            )

        else:
            self.plot_bands = None
            self.band_curve = None

        self.setStyleSheet(
            "background-color: black;"
        )

    def reset_meter(self):
        self.engine.reset()

    def update_gui(self):
        if not self.running:
            return

        latest = None

        while True:
            try:
                latest = (
                    self.processor.data_queue.get_nowait()
                )
            except queue.Empty:
                break

        if latest is None:
            return

        (
            timestamp,
            db_raw,
            db_weighted,
            fft_freq,
            fft_raw,
            fft_weighted,
            band_levels,
            leq_raw,
            leq_weighted,
        ) = latest

        self.value_raw.setText(
            f"Z: {db_raw:6.1f} dB"
        )

        self.value_weighted.setText(
            f"{self.engine.weighting_symbol if hasattr(self.engine, 'weighting_symbol') else self.engine.weighting_type}: "
            f"{db_weighted:6.1f} dB"
        )

        self.peak_raw_label.setText(
            f"Peak Z: "
            f"{self.engine.peak_raw:6.1f} dB"
        )

        self.peak_weighted_label.setText(
            f"Peak weighted: "
            f"{self.engine.peak_weighted:6.1f} dB"
        )

        self.leq_raw_label.setText(
            f"Leq Z: {leq_raw:6.1f} dB"
        )

        self.leq_weighted_label.setText(
            f"Leq weighted: "
            f"{leq_weighted:6.1f} dB"
        )

        if self.plot_time is not None:
            self.time_x.append(timestamp)
            self.time_raw.append(db_raw)
            self.time_weighted.append(
                db_weighted
            )

            if len(self.time_x) > self.max_time_points:
                self.time_x = self.time_x[
                    -self.max_time_points:
                ]

                self.time_raw = self.time_raw[
                    -self.max_time_points:
                ]

                self.time_weighted = (
                    self.time_weighted[
                        -self.max_time_points:
                    ]
                )

            self.curve_time_raw.setData(
                self.time_x,
                self.time_raw,
            )

            self.curve_time_weighted.setData(
                self.time_x,
                self.time_weighted,
            )

        if (
            fft_freq is not None
            and fft_raw is not None
            and self.plot_fft is not None
        ):
            mask = fft_freq > 0

            self.last_fft_freq = fft_freq[mask]
            self.last_fft_raw = fft_raw[mask]

            self.last_fft_weighted = (
                fft_weighted[mask]
                if fft_weighted is not None
                else None
            )

            self.curve_fft_raw.setData(
                self.last_fft_freq,
                self.last_fft_raw,
            )

            if self.last_fft_weighted is not None:
                self.curve_fft_weighted.setData(
                    self.last_fft_freq,
                    self.last_fft_weighted,
                )

        if (
            band_levels is not None
            and self.band_curve is not None
        ):
            centers = np.asarray(
                [
                    band["fc"]
                    for band in self.engine.bands
                ]
            )

            self.band_curve.setData(
                centers,
                band_levels,
            )

    def closeEvent(self, event):
        self.running = False
        self.processor.stop_event.set()
        event.accept()


# ============================================================
# MAIN
# ============================================================


def main():
    select_audio_devices()

    config = ask_configuration()

    # --------------------------------------------------------
    # Sursa audio
    # --------------------------------------------------------

    audio_data = None

    if config["source"] == "1":
        wav_fs, wav_data = wavfile.read(
            config["wav_path"]
        )

        audio_data = normalize_audio(
            wav_data
        )

        sample_rate = wav_fs

        print(
            f"\nWAV sample rate: "
            f"{sample_rate} Hz"
        )

    else:
        sample_rate = SAMPLE_RATE_DEFAULT

    print(
        f"Sample rate folosit: "
        f"{sample_rate} Hz"
    )

    # --------------------------------------------------------
    # Microphone frequency calibration
    # --------------------------------------------------------

    calibration_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        DEFAULT_CALIBRATION_FILE,
    )

    try:
        mic_calibration = (
            MicrophoneCalibration(
                calibration_path,
                enabled=MIC_CALIBRATION_ENABLED,
            )
        )

        print(
            "\n"
            + mic_calibration.description()
        )

        if mic_calibration.is_loaded:
            print(
                f"Fisier calibrare: "
                f"{calibration_path}"
            )

            print(
                "Exemple corectie:"
            )

            for frequency in (
                1000,
                5000,
                10000,
                15000,
                20000,
            ):
                correction = (
                    mic_calibration.correction_db(
                        frequency
                    )
                )

                print(
                    f"  {frequency:5d} Hz : "
                    f"{correction:+.2f} dB"
                )

    except Exception as exc:
        print(
            "\nEROARE calibrare microfon:"
        )
        print(exc)
        print(
            "Programul continua fara "
            "corectie de raspuns."
        )

        mic_calibration = (
            MicrophoneCalibration(
                enabled=False
            )
        )

    # --------------------------------------------------------
    # Engine
    # --------------------------------------------------------

    engine = MeterEngine(
        fs=sample_rate,
        mode=config["mode"],
        weighting_type=config["weighting_type"],
        band_fraction=config["band_fraction"],
        calibration_db=config["calibration_db"],
        microphone_calibration=mic_calibration,
    )

    # Needed by GUI labels.
    engine.weighting_symbol = (
        config["weighting_symbol"]
    )

    print(
        f"\nS-au generat "
        f"{len(engine.bands)} benzi "
        f"de 1/{config['band_fraction']} octava."
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = RealtimeProcessor(
        engine=engine,
        source=config["source"],
        audio_data=audio_data,
    )

    processing_thread = threading.Thread(
        target=processor.processing_loop,
        daemon=True,
    )

    processing_thread.start()

    # --------------------------------------------------------
    # Audio callback
    # --------------------------------------------------------

    play_pointer = {"value": 0}

    def audio_callback(
        indata,
        outdata,
        frames,
        status,
    ):
        if status:
            print(
                status,
                file=sys.stderr,
            )

        if config["source"] == "2":
            raw = (
                indata[:, 0]
                .astype(
                    np.float32,
                    copy=True,
                )
            )

            weighted = (
                engine.weighting_filter(
                    raw
                )
            )

            pointer = (
                play_pointer["value"]
            )

            play_pointer["value"] += len(
                raw
            )

            try:
                processor.raw_queue.put_nowait(
                    (
                        raw,
                        weighted.copy(),
                        pointer,
                    )
                )
            except queue.Full:
                pass

            outdata.fill(0)

            return

        start = play_pointer["value"]
        end = start + frames

        raw = audio_data[start:end]

        if len(raw) == 0:
            outdata.fill(0)
            raise sd.CallbackStop()

        weighted = (
            engine.weighting_filter(
                raw
            )
        )

        outdata.fill(0)
        outdata[:len(raw), 0] = weighted

        play_pointer["value"] += len(raw)

        try:
            processor.raw_queue.put_nowait(
                (
                    raw.astype(
                        np.float32,
                        copy=False,
                    ),
                    weighted.astype(
                        np.float32,
                        copy=False,
                    ),
                    play_pointer["value"],
                )
            )
        except queue.Full:
            pass

        if len(raw) < frames:
            raise sd.CallbackStop()

    # --------------------------------------------------------
    # Stream
    # --------------------------------------------------------

    if config["source"] == "2":
        stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=BLOCKSIZE,
            latency=LATENCY,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
    else:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=BLOCKSIZE,
            latency=LATENCY,
            channels=1,
            dtype="float32",
            callback=(
                lambda outdata,
                frames,
                time_info,
                status: audio_callback(
                    None,
                    outdata,
                    frames,
                    status,
                )
            ),
        )

    # --------------------------------------------------------
    # GUI
    # --------------------------------------------------------

    pg.setConfigOptions(
        antialias=False,
        background="k",
        foreground="w",
    )

    app = (
        QtWidgets.QApplication.instance()
        or QtWidgets.QApplication(sys.argv)
    )

    window = MeterWindow(
        engine,
        processor,
        config["graph_option"],
    )

    window.show()

    print(
        "\n=============================================="
    )
    print("Sonometrul ruleaza.")
    print(
        f"Mic calibration: "
        f"{mic_calibration.description()}"
    )
    print(
        "Apasa inchidere fereastra pentru stop."
    )
    print(
        "==============================================\n"
    )

    try:
        stream.start()
        app.exec()
    finally:
        processor.stop_event.set()

        try:
            stream.stop()
        except Exception:
            pass

        try:
            stream.close()
        except Exception:
            pass

        processing_thread.join(
            timeout=1.0
        )


if __name__ == "__main__":
    main()
