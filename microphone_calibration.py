"""
microphone_calibration.py

Frequency-response correction for the microphone calibration file.

The calibration file must contain two numeric columns:
    frequency_Hz    correction_dB

The correction values are the microphone response error. Therefore:
    corrected_dB = measured_dB - correction_dB

The correction is interpolated in log-frequency, which is appropriate for
audio-frequency response curves and octave-band analysis.
"""

from pathlib import Path
import numpy as np


class MicrophoneCalibration:
    def __init__(self, calibration_file=None, enabled=True):
        self.enabled = bool(enabled)
        self.file_path = Path(calibration_file) if calibration_file else None

        self.frequencies = np.array([], dtype=np.float64)
        self.corrections_db = np.array([], dtype=np.float64)

        if self.enabled and self.file_path:
            self.load(self.file_path)

    def load(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {path}"
            )

        frequencies = []
        corrections = []

        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.replace(",", ".").split()

                if len(parts) < 2:
                    continue

                try:
                    frequency = float(parts[0])
                    correction = float(parts[1])
                except ValueError:
                    continue

                if frequency <= 0:
                    continue

                frequencies.append(frequency)
                corrections.append(correction)

        if len(frequencies) < 2:
            raise ValueError(
                f"Calibration file contains fewer than two valid points: {path}"
            )

        frequencies = np.asarray(frequencies, dtype=np.float64)
        corrections = np.asarray(corrections, dtype=np.float64)

        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        corrections = corrections[order]

        unique_frequencies, unique_indices = np.unique(
            frequencies, return_index=True
        )

        self.frequencies = unique_frequencies
        self.corrections_db = corrections[unique_indices]
        self.file_path = path

    @property
    def is_loaded(self):
        return (
            self.enabled
            and len(self.frequencies) >= 2
            and len(self.corrections_db) >= 2
        )

    def correction_db(self, frequency_hz):
        """
        Return microphone correction in dB.

        The calibration curve is interpolated linearly in log10(frequency).
        Outside the calibration range the nearest endpoint is used.
        """
        if not self.is_loaded:
            value = np.asarray(frequency_hz, dtype=np.float64)
            return np.zeros_like(value) if value.ndim else 0.0

        f = np.asarray(frequency_hz, dtype=np.float64)
        safe_f = np.maximum(f, self.frequencies[0])

        correction = np.interp(
            np.log10(safe_f),
            np.log10(self.frequencies),
            self.corrections_db,
            left=self.corrections_db[0],
            right=self.corrections_db[-1],
        )

        if np.ndim(frequency_hz) == 0:
            return float(correction)

        return correction

    def correct_db(self, measured_db, frequency_hz):
        """Apply microphone response correction to an amplitude level."""
        return np.asarray(measured_db) - self.correction_db(frequency_hz)

    def correct_fft_db(self, fft_db, frequencies_hz):
        """Apply the response correction to an FFT spectrum in dB."""
        return np.asarray(fft_db) - self.correction_db(frequencies_hz)

    def correct_band_db(self, band_db, center_frequencies_hz):
        """Apply the response correction to octave-band levels."""
        return np.asarray(band_db) - self.correction_db(center_frequencies_hz)

    def description(self):
        if not self.is_loaded:
            return "Mic Cal: OFF"

        return (
            f"Mic Cal: ON "
            f"({self.frequencies[0]:.0f}-{self.frequencies[-1]:.0f} Hz)"
        )
