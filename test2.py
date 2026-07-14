import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def desenare(path):
    sample_rate, audio_data = wavfile.read(path)
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Tipul de date al semnalului audio: {audio_data.dtype}")
    print(f"dimensiunea semnalului audio: {audio_data.shape}")
    if len(audio_data.shape) > 1:
        print(f"Numărul de canale: {audio_data.shape[1]}")
        audio_data = audio_data[:, 0] 

    #normalizarea semnalului audio
    if audio_data.dtype == np.int16:
        semnal_normalizat = audio_data / 32768.0
    elif audio_data.dtype == np.int32:
        semnal_normalizat = audio_data / 2147483648.0
    else:
        semnal_normalizat = audio_data

    number_of_samples = len(semnal_normalizat)
    duration = number_of_samples / sample_rate
    timp = np.arange(0, number_of_samples) / sample_rate
    plt.figure(figsize=(10, 4))
    esantioane_de_afisat=int(sample_rate)
    plt.plot(timp[:esantioane_de_afisat], semnal_normalizat[:esantioane_de_afisat])
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude(normalizat)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    path = input("Introduceți calea către fișierul WAV: ")
    desenare(path)
