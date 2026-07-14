import numpy as np
from scipy.io import wavfile

# Parametrii semnalului
fs = 44100              # Rata de eșantionare (44.1 kHz)
durata_acord = 5.0      # Fiecare acord cântă 5 secunde
durata_liniste = 2.0    # Liniște la final

# Generăm axa de timp pentru o singură bucată de acord (5 secunde)
t = np.linspace(0, durata_acord, int(fs * durata_acord), endpoint=False)

# Definim progresia de acorduri cu frecvențele lor
progresie_acorduri = [
    {"nume": "Am (I)",    "frecvente": [220.00, 261.63, 329.63]},  # A3, C4, E4
    {"nume": "Bdim (II)", "frecvente": [246.94, 293.66, 349.23]},  # B3, D4, F4
    {"nume": "Cmaj (III)", "frecvente": [261.63, 329.63, 392.00]},  # C4, E4, G4
    {"nume": "Dm (IV)",   "frecvente": [293.66, 349.23, 440.00]},  # D4, F4, A4
    {"nume": "Em (V)",    "frecvente": [329.63, 392.00, 493.88]},  # E4, G4, B4
    {"nume": "Fmaj (VI)", "frecvente": [349.23, 440.00, 523.25]},  # F4, A4, C5
    {"nume": "Gmaj (VII)","frecvente": [392.00, 493.88, 587.33]},  # G4, B4, D5
    {"nume": "Am (I - Octava 4)", "frecvente": [440.00, 523.25, 659.25]}  # A4, C5, E5
]

semnal_complet = np.array([])

print("Generăm progresia de acorduri pentru gama Am...")

# Buclă prin fiecare acord din progresie
for acord in progresie_acorduri:
    print(f" - Generăm {acord['nume']}: {acord['frecvente']} Hz")
    
    # Generăm acordul curent adunând cele 3 sinusoide
    sunet_acord = np.zeros_like(t)
    for f in acord['frecvente']:
        sunet_acord += np.sin(2 * np.pi * f * t)
        
    # Normalizăm acordul curent individual pentru a evita clipping-ul
    sunet_acord = sunet_acord / np.max(np.abs(sunet_acord))
    
    # Lipim acordul curent la semnalul complet
    semnal_complet = np.concatenate((semnal_complet, sunet_acord))

# Adăugăm cele 2 secunde de liniște la finalul întregii progresii
esantioane_liniste = int(fs * durata_liniste)
semnal_liniste = np.zeros(esantioane_liniste)
semnal_complet = np.concatenate((semnal_complet, semnal_liniste))

# Convertim semnalul la formatul final 16-bit PCM pentru fișierul WAV
semnal_int16 = (semnal_complet * 32767).astype(np.int16)

# Salvăm fișierul
wavfile.write("test.wav", fs, semnal_int16)
print("\nFișierul 'test.wav' a fost generat cu succes! Durată totală: 42 de secunde.")