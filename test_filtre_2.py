import os
import queue
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import (
    bilinear_zpk,
    butter,
    sosfilt,
    sosfilt_zi,
    zpk2sos,
)

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.io.wavfile")

try:
    import sounddevice as sd
except ImportError:
    print("Lipsește sounddevice. Instalează: pip install sounddevice")
    sys.exit(1)

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("Lipsesc pyqtgraph/PyQt5. Instalează: pip install pyqtgraph pyqt5")
    sys.exit(1)


# ============================================================
# CONSTANTE ȘI CONFIGURARE
# ============================================================

EPSILON = 1e-20
REFERENCE_PRESSURE_PA = 20e-6
MIN_ANALYSIS_HZ = 20.0
MAX_ANALYSIS_HZ = 20_000.0
DEFAULT_SAMPLE_RATE = 48_000
GUI_UPDATE_MS = 33
FFT_POINTS_MAX = 4096
HISTORY_SECONDS = 30.0
FILTER_ORDER = 6

# IEC 61260 folosește raportul de octavă exact G = 10^(3/10).
OCTAVE_RATIO = 10.0 ** (3.0 / 10.0)

COL_RAW = (255, 165, 0)
COL_WEIGHTED = (0, 220, 220)
COL_BARS = (80, 180, 255)
COL_GRID = (100, 100, 100)


@dataclass
class RuntimeConfig:
    source: str
    wav_path: Optional[Path]
    sample_rate: int
    input_device: Optional[int]
    output_device: Optional[int]
    weighting: str
    fraction: int
    time_weighting: str
    calibration_offset_db: Optional[float]

    @property
    def calibrated(self) -> bool:
        return self.calibration_offset_db is not None

    @property
    def level_unit(self) -> str:
        if self.calibrated:
            return f"dB {self.weighting} SPL"
        return f"dB{self.weighting}FS"


# ============================================================
# UTILITARE DE CONFIGURARE
# ============================================================


def read_choice(prompt: str, valid: dict[str, str], default_key: str) -> str:
    while True:
        value = input(prompt).strip().lower()
        if not value:
            value = default_key
        if value in valid:
            return valid[value]
        print("Opțiune invalidă. Încearcă din nou.")


def select_audio_devices() -> tuple[Optional[int], Optional[int]]:
    devices = sd.query_devices()

    print("\n~~~~~ DISPOZITIVE AUDIO DE INTRARE ~~~~~")
    input_indices = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            input_indices.append(index)
            print(f"[{index}] {device['name']} ({device['max_input_channels']} canale)")

    input_device = None
    value = input("ID intrare [Enter = implicit]: ").strip()
    if value:
        try:
            candidate = int(value)
            if candidate in input_indices:
                input_device = candidate
            else:
                print("ID de intrare invalid; se folosește dispozitivul implicit.")
        except ValueError:
            print("Valoare invalidă; se folosește dispozitivul implicit.")

    print("\n~~~~~ DISPOZITIVE AUDIO DE IEȘIRE ~~~~~")
    output_indices = []
    for index, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            output_indices.append(index)
            print(f"[{index}] {device['name']} ({device['max_output_channels']} canale)")

    output_device = None
    value = input("ID ieșire [Enter = implicit]: ").strip()
    if value:
        try:
            candidate = int(value)
            if candidate in output_indices:
                output_device = candidate
            else:
                print("ID de ieșire invalid; se folosește dispozitivul implicit.")
        except ValueError:
            print("Valoare invalidă; se folosește dispozitivul implicit.")

    return input_device, output_device


def build_config() -> RuntimeConfig:
    print("\n~~~~~ CONFIGURARE SURSĂ ~~~~~")
    print("1. Fișier WAV")
    print("2. Microfon live")
    source = read_choice("> ", {"1": "wav", "2": "live"}, "2")

    wav_path = None
    sample_rate = DEFAULT_SAMPLE_RATE
    if source == "wav":
        wav_path = Path(input("Calea fișierului WAV: ").strip().strip('"'))
        if not wav_path.exists():
            raise FileNotFoundError(f"Fișierul nu există: {wav_path}")

    print("\n~~~~~ PONDERARE ÎN FRECVENȚĂ ~~~~~")
    print("1. A-weighting")
    print("2. C-weighting")
    print("3. Z-weighting (semnal nemodificat)")
    weighting = read_choice(
        "> ",
        {
            "1": "A",
            "a": "A",
            "2": "C",
            "c": "C",
            "3": "Z",
            "z": "Z",
        },
        "1",
    )

    print("\n~~~~~ ANALIZĂ PE BENZI ~~~~~")
    print("1. Benzi de 1 octavă")
    print("2. Benzi de 1/3 octavă")
    print("3. Benzi de 1/6 octavă")
    print("4. Benzi de 1/12 octavă")
    fraction = int(
        read_choice(
            "> ",
            {"1": "1", "2": "3", "3": "6", "4": "12"},
            "2",
        )
    )

    print("\n~~~~~ PONDERARE TEMPORALĂ ~~~~~")
    print("1. FAST (τ = 125 ms)")
    print("2. SLOW (τ = 1 s)")
    time_weighting = read_choice(
        "> ",
        {"1": "FAST", "fast": "FAST", "2": "SLOW", "slow": "SLOW"},
        "1",
    )

    print("\n~~~~~ CALIBRARE ~~~~~")
    print("Pentru dB SPL real este necesară calibrarea lanțului microfon + interfață.")
    print("Introdu offsetul: nivel_SPL_cunoscut - nivel_dBFS_măsurat.")
    print("Exemplu: calibrator 94 dB, citire -26 dBFS => offset 120 dB.")
    value = input("Offset calibrare în dB [Enter = afișare dBFS]: ").strip()
    calibration_offset_db = None
    if value:
        try:
            calibration_offset_db = float(value)
        except ValueError as exc:
            raise ValueError("Offsetul de calibrare trebuie să fie numeric.") from exc

    input_device, output_device = select_audio_devices()

    return RuntimeConfig(
        source=source,
        wav_path=wav_path,
        sample_rate=sample_rate,
        input_device=input_device,
        output_device=output_device,
        weighting=weighting,
        fraction=fraction,
        time_weighting=time_weighting,
        calibration_offset_db=calibration_offset_db,
    )


# ============================================================
# FILTRE DE PONDERARE A / C / Z
# ============================================================


def a_weighting_sos(sample_rate: float) -> np.ndarray:
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = -2.0

    zeros = [0.0, 0.0, 0.0, 0.0]
    poles = [
        -2.0 * np.pi * f1,
        -2.0 * np.pi * f1,
        -2.0 * np.pi * f2,
        -2.0 * np.pi * f3,
        -2.0 * np.pi * f4,
        -2.0 * np.pi * f4,
    ]
    gain = (2.0 * np.pi * f4) ** 2 * 10.0 ** (a1000 / 20.0)
    zd, pd, kd = bilinear_zpk(zeros, poles, gain, sample_rate)
    return zpk2sos(zd, pd, kd)


def c_weighting_sos(sample_rate: float) -> np.ndarray:
    f1, f4 = 20.598997, 12194.217
    c1000 = -0.0619

    zeros = [0.0, 0.0]
    poles = [
        -2.0 * np.pi * f1,
        -2.0 * np.pi * f1,
        -2.0 * np.pi * f4,
        -2.0 * np.pi * f4,
    ]
    gain = (2.0 * np.pi * f4) ** 2 * 10.0 ** (c1000 / 20.0)
    zd, pd, kd = bilinear_zpk(zeros, poles, gain, sample_rate)
    return zpk2sos(zd, pd, kd)


class FrequencyWeightingFilter:
    def __init__(self, weighting: str, sample_rate: float):
        self.weighting = weighting
        self.sample_rate = sample_rate
        self.sos: Optional[np.ndarray]
        self.zi: Optional[np.ndarray]

        if weighting == "A":
            self.sos = a_weighting_sos(sample_rate)
        elif weighting == "C":
            self.sos = c_weighting_sos(sample_rate)
        elif weighting == "Z":
            self.sos = None
        else:
            raise ValueError(f"Ponderare necunoscută: {weighting}")

        self.zi = sosfilt_zi(self.sos) * 0.0 if self.sos is not None else None

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.sos is None:
            return samples.copy()
        filtered, self.zi = sosfilt(self.sos, samples, zi=self.zi)
        return filtered


# ============================================================
# BANCĂ PARALELĂ DE FILTRE FRACȚIONARE DE OCTAVĂ
# ============================================================


def generate_exact_band_centers(
    fraction: int,
    minimum_hz: float,
    maximum_hz: float,
) -> np.ndarray:
    """Centre exacte cu referință 1000 Hz și raport G = 10^(3/10)."""
    centers = []
    index = -300
    while index <= 300:
        center = 1000.0 * OCTAVE_RATIO ** (index / fraction)
        lower = center * OCTAVE_RATIO ** (-1.0 / (2.0 * fraction))
        upper = center * OCTAVE_RATIO ** (1.0 / (2.0 * fraction))
        if lower >= minimum_hz and upper <= maximum_hz:
            centers.append(center)
        index += 1
    return np.asarray(centers, dtype=np.float64)


def format_center_frequency(frequency: float) -> str:
    if frequency >= 1000.0:
        value = frequency / 1000.0
        if value >= 10:
            return f"{value:.0f}k"
        return f"{value:.1f}k".replace(".0k", "k")
    if frequency >= 100:
        return f"{frequency:.0f}"
    return f"{frequency:.1f}".replace(".0", "")


class FractionalOctaveFilterBank:
    """
    Bancă de filtre IIR Butterworth procesate în paralel.

    Centrele și limitele geometrice urmează convenția IEC 61260.
    Conformitatea completă cu măștile de toleranță IEC necesită validare
    metrologică și nu poate fi garantată doar prin acest cod.
    """

    def __init__(self, sample_rate: float, fraction: int, order: int = FILTER_ORDER):
        self.sample_rate = sample_rate
        self.fraction = fraction
        self.order = order

        maximum = min(MAX_ANALYSIS_HZ, sample_rate * 0.49)
        self.centers = generate_exact_band_centers(fraction, MIN_ANALYSIS_HZ, maximum)
        if self.centers.size == 0:
            raise ValueError("Nu există benzi valide pentru rata de eșantionare selectată.")

        self.sos_filters: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

        for center in self.centers:
            lower = center * OCTAVE_RATIO ** (-1.0 / (2.0 * fraction))
            upper = center * OCTAVE_RATIO ** (1.0 / (2.0 * fraction))
            sos = butter(
                order,
                [lower, upper],
                btype="bandpass",
                fs=sample_rate,
                output="sos",
            )
            self.sos_filters.append(sos)
            self.states.append(sosfilt_zi(sos) * 0.0)

    def process(self, samples: np.ndarray) -> np.ndarray:
        outputs = np.empty((len(self.sos_filters), len(samples)), dtype=np.float64)
        for index, sos in enumerate(self.sos_filters):
            outputs[index], self.states[index] = sosfilt(
                sos,
                samples,
                zi=self.states[index],
            )
        return outputs


# ============================================================
# NIVELURI RMS CU PONDERARE TEMPORALĂ
# ============================================================


class ExponentialLevelDetector:
    def __init__(self, sample_rate: float, tau_seconds: float, channels: int = 1):
        self.alpha = float(np.exp(-1.0 / (sample_rate * tau_seconds)))
        self.power = np.full(channels, EPSILON, dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Primește [frames] sau [channels, frames] și returnează puterea finală."""
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]

        one_minus_alpha = 1.0 - self.alpha
        for frame_index in range(data.shape[1]):
            self.power = self.alpha * self.power + one_minus_alpha * np.square(
                data[:, frame_index]
            )
        return self.power.copy()


def power_to_db(power: np.ndarray | float, calibration_offset_db: Optional[float]) -> np.ndarray:
    level = 10.0 * np.log10(np.maximum(power, EPSILON))
    if calibration_offset_db is not None:
        level = level + calibration_offset_db
    return np.asarray(level)


# ============================================================
# MOTOR DSP
# ============================================================


class SoundLevelProcessor:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        tau = 0.125 if config.time_weighting == "FAST" else 1.0

        self.weighting_filter = FrequencyWeightingFilter(config.weighting, config.sample_rate)
        self.filter_bank = FractionalOctaveFilterBank(
            config.sample_rate,
            config.fraction,
        )
        self.overall_detector = ExponentialLevelDetector(config.sample_rate, tau, channels=1)
        self.raw_detector = ExponentialLevelDetector(config.sample_rate, tau, channels=1)
        self.band_detector = ExponentialLevelDetector(
            config.sample_rate,
            tau,
            channels=len(self.filter_bank.centers),
        )

        self.fft_size = min(FFT_POINTS_MAX, int(config.sample_rate))
        self.fft_buffer_raw = np.zeros(self.fft_size, dtype=np.float64)
        self.fft_buffer_weighted = np.zeros(self.fft_size, dtype=np.float64)
        self.fft_window = np.hanning(self.fft_size)
        self.fft_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / config.sample_rate)

    @staticmethod
    def update_ring(buffer: np.ndarray, samples: np.ndarray) -> None:
        count = len(samples)
        if count >= len(buffer):
            buffer[:] = samples[-len(buffer):]
        else:
            buffer[:-count] = buffer[count:]
            buffer[-count:] = samples

    def calculate_fft(self, buffer: np.ndarray) -> np.ndarray:
        coherent_gain = np.sum(self.fft_window) / 2.0
        spectrum = np.abs(np.fft.rfft(buffer * self.fft_window)) / max(coherent_gain, EPSILON)
        db = 20.0 * np.log10(np.maximum(spectrum, EPSILON))
        if self.config.calibration_offset_db is not None:
            db += self.config.calibration_offset_db
        return db

    def process(self, raw_samples: np.ndarray) -> dict[str, np.ndarray | float]:
        weighted = self.weighting_filter.process(raw_samples)
        band_signals = self.filter_bank.process(weighted)

        raw_power = self.raw_detector.process(raw_samples)[0]
        overall_power = self.overall_detector.process(weighted)[0]
        band_power = self.band_detector.process(band_signals)

        self.update_ring(self.fft_buffer_raw, raw_samples)
        self.update_ring(self.fft_buffer_weighted, weighted)

        return {
            "weighted_audio": weighted,
            "raw_level": float(power_to_db(raw_power, self.config.calibration_offset_db)),
            "overall_level": float(
                power_to_db(overall_power, self.config.calibration_offset_db)
            ),
            "band_levels": power_to_db(
                band_power,
                self.config.calibration_offset_db,
            ),
            "fft_raw": self.calculate_fft(self.fft_buffer_raw),
            "fft_weighted": self.calculate_fft(self.fft_buffer_weighted),
        }


# ============================================================
# INTERFAȚĂ GRAFICĂ
# ============================================================


class SoundMeterWindow(QtWidgets.QWidget):
    def __init__(self, config: RuntimeConfig, processor: SoundLevelProcessor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.data_queue: queue.Queue = queue.Queue(maxsize=8)
        self.running = True
        self.stream = None
        self.play_pointer = 0
        self.wav_audio: Optional[np.ndarray] = None
        self.history_t: list[float] = []
        self.history_raw: list[float] = []
        self.history_weighted: list[float] = []
        self.peak_level = -np.inf

        self.setWindowTitle(
            f"Sonometru software – {config.weighting}/{config.time_weighting} – "
            f"1/{config.fraction} octavă"
        )
        self.resize(1500, 900)
        self.setStyleSheet("background-color: black; color: white;")

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(GUI_UPDATE_MS)

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, stretch=5)

        level_title = "Nivel temporal"
        self.level_plot = self.graphics.addPlot(title=level_title)
        self.level_plot.setLabel("bottom", "Timp", units="s")
        self.level_plot.setLabel("left", self.config.level_unit)
        self.level_plot.showGrid(x=True, y=True, alpha=0.25)
        self.level_plot.addLegend()
        self.raw_curve = self.level_plot.plot(
            pen=pg.mkPen(COL_RAW, width=1.5),
            name="Z / intrare brută",
        )
        self.weighted_curve = self.level_plot.plot(
            pen=pg.mkPen(COL_WEIGHTED, width=2.0),
            name=f"{self.config.weighting}-{self.config.time_weighting}",
        )

        self.graphics.nextRow()
        self.fft_plot = self.graphics.addPlot(title="Spectru FFT instantaneu")
        self.fft_plot.setLabel("bottom", "Frecvență", units="Hz")
        self.fft_plot.setLabel("left", "Nivel spectral", units="dB")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fft_plot.addLegend()
        self.fft_raw_curve = self.fft_plot.plot(
            pen=pg.mkPen(COL_RAW, width=1.0),
            name="FFT intrare",
        )
        self.fft_weighted_curve = self.fft_plot.plot(
            pen=pg.mkPen(COL_WEIGHTED, width=1.5),
            name=f"FFT {self.config.weighting}-weighted",
        )
        max_frequency = min(MAX_ANALYSIS_HZ, self.config.sample_rate / 2.0)
        self.fft_plot.setXRange(np.log10(MIN_ANALYSIS_HZ), np.log10(max_frequency))

        self.graphics.nextRow()
        self.band_plot = self.graphics.addPlot(
            title=f"Spectru pe benzi de 1/{self.config.fraction} octavă"
        )
        self.band_plot.setLabel("bottom", "Frecvența centrală", units="Hz")
        self.band_plot.setLabel("left", self.config.level_unit)
        self.band_plot.showGrid(x=False, y=True, alpha=0.25)

        centers = self.processor.filter_bank.centers
        self.band_x = np.arange(len(centers), dtype=np.float64)
        self.band_width = 0.8
        self.band_bars = pg.BarGraphItem(
            x=self.band_x,
            height=np.full(len(centers), -120.0),
            width=self.band_width,
            brush=COL_BARS,
        )
        self.band_plot.addItem(self.band_bars)

        # Afișăm un număr controlat de etichete pentru lizibilitate.
        label_step = max(1, len(centers) // 24)
        ticks = [
            (float(index), format_center_frequency(centers[index]))
            for index in range(0, len(centers), label_step)
        ]
        self.band_plot.getAxis("bottom").setTicks([ticks])
        self.band_plot.setXRange(-1, len(centers))

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel, stretch=1)

        self.mode_label = QtWidgets.QLabel(
            f"{self.config.weighting}-weighting\n"
            f"{self.config.time_weighting}\n"
            f"1/{self.config.fraction} octavă"
        )
        self.mode_label.setAlignment(QtCore.Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        panel_layout.addWidget(self.mode_label)

        self.level_lcd = QtWidgets.QLCDNumber()
        self.level_lcd.setDigitCount(6)
        self.level_lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.level_lcd.setMinimumHeight(120)
        self.level_lcd.setStyleSheet(
            "QLCDNumber { color: #39ff14; background-color: #111; border: 1px solid #555; }"
        )
        panel_layout.addWidget(self.level_lcd)

        self.unit_label = QtWidgets.QLabel(self.config.level_unit)
        self.unit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.unit_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        panel_layout.addWidget(self.unit_label)

        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setOrientation(QtCore.Qt.Vertical)
        self.level_bar.setTextVisible(False)
        self.level_bar.setRange(0, 1400)
        self.level_bar.setMinimumHeight(400)
        self.level_bar.setStyleSheet(
            "QProgressBar { background: #181818; border: 1px solid #555; }"
            "QProgressBar::chunk { background: #39ff14; }"
        )
        panel_layout.addWidget(self.level_bar, alignment=QtCore.Qt.AlignHCenter)

        self.peak_label = QtWidgets.QLabel("Max: --")
        self.peak_label.setAlignment(QtCore.Qt.AlignCenter)
        self.peak_label.setStyleSheet("font-size: 16px;")
        panel_layout.addWidget(self.peak_label)

        self.calibration_label = QtWidgets.QLabel()
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setAlignment(QtCore.Qt.AlignCenter)
        if self.config.calibrated:
            self.calibration_label.setText(
                f"Calibrat software cu offset {self.config.calibration_offset_db:+.2f} dB."
            )
            self.calibration_label.setStyleSheet("color: #f1c40f;")
        else:
            self.calibration_label.setText(
                "NECALIBRAT: valorile sunt raportate la full scale, nu reprezintă dB SPL."
            )
            self.calibration_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        panel_layout.addWidget(self.calibration_label)

        self.reset_button = QtWidgets.QPushButton("Reset nivel maxim")
        self.reset_button.clicked.connect(self.reset_peak)
        panel_layout.addWidget(self.reset_button)
        panel_layout.addStretch(1)

    def reset_peak(self) -> None:
        self.peak_level = -np.inf
        self.peak_label.setText("Max: --")

    def push_result(self, timestamp: float, result: dict) -> None:
        item = (timestamp, result)
        try:
            self.data_queue.put_nowait(item)
        except queue.Full:
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
            self.data_queue.put_nowait(item)

    def process_audio(self, raw: np.ndarray) -> np.ndarray:
        result = self.processor.process(raw)
        timestamp = self.play_pointer / self.config.sample_rate
        self.push_result(timestamp, result)
        return np.asarray(result["weighted_audio"], dtype=np.float64)

    def playback_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        if self.wav_audio is None:
            outdata.fill(0)
            raise sd.CallbackStop()

        chunk = self.wav_audio[self.play_pointer:self.play_pointer + frames]
        valid_frames = len(chunk)
        if valid_frames == 0:
            outdata.fill(0)
            raise sd.CallbackStop()

        weighted = self.process_audio(chunk)
        outdata.fill(0)
        outdata[:valid_frames, 0] = weighted
        self.play_pointer += valid_frames

        if valid_frames < frames:
            raise sd.CallbackStop()

    def record_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        raw = indata[:, 0].astype(np.float64, copy=True)
        self.process_audio(raw)
        self.play_pointer += len(raw)

    def level_to_bar_value(self, level: float) -> int:
        if self.config.calibrated:
            # Scală practică 0...140 dB SPL.
            return int(np.clip(level * 10.0, 0.0, 1400.0))
        # Scală dBFS -120...0 mapată pe 0...1200.
        return int(np.clip((level + 120.0) * 10.0, 0.0, 1200.0))

    def update_gui(self) -> None:
        latest = None
        while True:
            try:
                latest = self.data_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            if self.stream is not None and not self.stream.active:
                self.close()
            return

        timestamp, result = latest
        raw_level = float(result["raw_level"])
        overall_level = float(result["overall_level"])
        band_levels = np.asarray(result["band_levels"])

        self.history_t.append(timestamp)
        self.history_raw.append(raw_level)
        self.history_weighted.append(overall_level)

        cutoff = timestamp - HISTORY_SECONDS
        first_valid = 0
        while first_valid < len(self.history_t) and self.history_t[first_valid] < cutoff:
            first_valid += 1
        if first_valid:
            del self.history_t[:first_valid]
            del self.history_raw[:first_valid]
            del self.history_weighted[:first_valid]

        self.raw_curve.setData(self.history_t, self.history_raw)
        self.weighted_curve.setData(self.history_t, self.history_weighted)
        if timestamp > HISTORY_SECONDS:
            self.level_plot.setXRange(timestamp - HISTORY_SECONDS, timestamp, padding=0)

        frequencies = self.processor.fft_freqs
        valid = (frequencies >= MIN_ANALYSIS_HZ) & (
            frequencies <= min(MAX_ANALYSIS_HZ, self.config.sample_rate / 2.0)
        )
        self.fft_raw_curve.setData(frequencies[valid], np.asarray(result["fft_raw"])[valid])
        self.fft_weighted_curve.setData(
            frequencies[valid],
            np.asarray(result["fft_weighted"])[valid],
        )

        baseline = 0.0 if self.config.calibrated else -120.0
        heights = band_levels - baseline
        self.band_bars.setOpts(
            x=self.band_x,
            y0=np.full(len(self.band_x), baseline),
            height=heights,
            width=self.band_width,
        )

        self.level_lcd.display(f"{overall_level:.1f}")
        self.level_bar.setValue(self.level_to_bar_value(overall_level))

        if overall_level > self.peak_level:
            self.peak_level = overall_level
        self.peak_label.setText(f"Max: {self.peak_level:.1f} {self.config.level_unit}")

    def closeEvent(self, event) -> None:
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        event.accept()

    def start(self) -> None:
        sd.default.device = (self.config.input_device, self.config.output_device)

        if self.config.source == "wav":
            sample_rate, audio = wavfile.read(str(self.config.wav_path))
            self.config.sample_rate = int(sample_rate)
            if audio.ndim > 1:
                audio = audio[:, 0]

            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                scale = max(abs(info.min), info.max)
                audio = audio.astype(np.float64) / scale
            else:
                audio = audio.astype(np.float64)

            self.wav_audio = audio
            self.stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                callback=self.playback_callback,
                device=self.config.output_device,
            )
        else:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                callback=self.record_callback,
                device=self.config.input_device,
            )

        self.stream.start()
        self.show()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    try:
        config = build_config()

        # Pentru WAV rata reală trebuie citită înainte de construirea filtrelor.
        if config.source == "wav" and config.wav_path is not None:
            sample_rate, _ = wavfile.read(str(config.wav_path), mmap=True)
            config.sample_rate = int(sample_rate)

        print("\n~~~~~ CONFIGURAȚIE ACTIVĂ ~~~~~")
        print(f"Ponderare: {config.weighting}")
        print(f"Temporal: {config.time_weighting}")
        print(f"Benzi: 1/{config.fraction} octavă")
        print(f"Rată eșantionare: {config.sample_rate} Hz")
        if config.calibrated:
            print(f"Offset calibrare: {config.calibration_offset_db:+.2f} dB")
        else:
            print("Calibrare: absentă; nivelurile vor fi afișate în dBFS.")

        processor = SoundLevelProcessor(config)
        print(f"Număr de filtre procesate în paralel: {len(processor.filter_bank.centers)}")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        window = SoundMeterWindow(config, processor)
        window.start()
        app.exec_()

    except KeyboardInterrupt:
        print("\nOprit de utilizator.")
    except Exception as exc:
        print(f"\nEroare: {exc}", file=sys.stderr)
        raise
    finally:
        sd.stop()


if __name__ == "__main__":
    main()