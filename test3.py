import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import time

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Module 'sounddevice' not found. Audio playback will be disabled.")

def realtime_spectrum(file_path):
    fs, audio_data = wavfile.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Use only the first channel if stereo
    
    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
    elif audio_data.dtype == np.int32:
        audio_data = audio_data / 2147483648.0  # Normalize to [-1, 1]

    window_size = 2048
    hop_size = 512
    durata_cadru_secunde = hop_size / fs

    fereastra = np.hanning(window_size)
    frecvente = np.fft.rfftfreq(window_size, d=1/fs)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    line, = ax.plot(frecvente, np.zeros_like(frecvente))
    ax.set_title("Spectru în timp real")
    ax.set_xlabel("Frecvență (Hz)")
    ax.set_ylabel("mag (dB)")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-60, 10)
    ax.grid(True)

    if HAS_SOUNDDEVICE:
        sd.play(audio_data, fs)

    timp_start = time.time()
    numar_esantioane = len(audio_data)
    index_curent=0

    print("Redarea și afișarea spectrului în timp real...") 
    
    try:
        while index_curent + window_size <= numar_esantioane:
            if HAS_SOUNDDEVICE:
                timp_scurs = time.time() - timp_start
                index_curent = int(timp_scurs * fs)
                if index_curent + window_size > numar_esantioane:
                    break
            else:
                index_curent += hop_size
                time.sleep(durata_cadru_secunde)
            chunk = audio_data[index_curent:index_curent + window_size]
            chunk_fereastruit = chunk * fereastra
            spectru = np.fft.rfft(chunk_fereastruit)
            magnitudine = np.abs(spectru)/window_size
            magnitudine_db = 20 * np.log10(magnitudine + 1e-10)
            line.set_ydata(magnitudine_db)

            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Redarea a fost întreruptă de utilizator.")
    finally:
        if HAS_SOUNDDEVICE:
            sd.stop()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    file_path = "test.wav"  #input("Introduceți calea către fișierul WAV: ")
    realtime_spectrum(file_path)