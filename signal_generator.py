import math
import struct
import wave

# Configurări fișier audio
filename = "stereo_test.wav"
duration = 12.0  # Durata în secunde (între 10 și 15)
sample_rate = 44100  # Calitate CD (44.1 kHz)
num_channels = 2  # Stereo (1 = Stânga, 2 = Dreapta)
sample_width = 2  # 2 bytes = 16-bit audio

# Parametri audio pentru fiecare canal
# Canal Stânga (Left)
freq_left = 220.0  # Frecvență joasă (Nota La/A3)
volume_left = 0.3  # Volum la 30% din capacitate

# Canal Dreapta (Right)
freq_right = 440.0  # Frecvență mai înaltă (Nota La/A4)
volume_right = 0.9  # Volum la 90% din capacitate (mult mai puternic)

total_samples = int(sample_rate * duration)
max_amplitude = 32767  # Valoarea maximă pentru audio pe 16-bit (signed short)

print(f"Se generează fișierul '{filename}'...")

# Deschidem fișierul WAV pentru scriere
with wave.open(filename, "w") as wav_file:
    # Setăm parametrii: (nchannels, sampwidth, framerate, nframes, comptype, compname)
    wav_file.setparams(
        (num_channels, sample_width, sample_rate, total_samples, "NONE", "not compressed")
    )

    audio_frames = bytearray()

    for i in range(total_samples):
        # Timpul curent în secunde
        t = i / sample_rate

        # Generăm unda sinusoidală pentru canalul stânga
        sample_left = math.sin(2 * math.pi * freq_left * t)
        val_left = int(sample_left * max_amplitude * volume_left)

        # Generăm unda sinusoidală pentru canalul dreapta
        sample_right = math.sin(2 * math.pi * freq_right * t)
        val_right = int(sample_right * max_amplitude * volume_right)

        # Împachetăm valorile în format binar (short pe 16 biți: 'h')
        # Formatul stereo în WAV pune eșantioanele alternativ: Stânga, Dreapta, Stânga, Dreapta...
        packed_left = struct.pack("<h", val_left)
        packed_right = struct.pack("<h", val_right)

        audio_frames.extend(packed_left)
        audio_frames.extend(packed_right)

    # Scriem toate datele în fișier
    wav_file.writeframes(audio_frames)

print("Fișierul a fost generat cu succes!")