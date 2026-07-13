import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def simulator_sonometru_absolut(cale_fisier, mod_timp="Fast", c_calib=10.0):
    """Simulează un sonometru conform formulelor din capitolul 1 și 2.

    c_calib: Presiunea maximă în Pascali (Pa) corespunzătoare valorii digitale
    1.0.
    """
    # Pasul 1: Citirea semnalului stereo (Capitolul 2.1)
    rate, data = wavfile.read("wavv.wav")

    # Convertim datele brute la amplitudine normată digital (între -1.0 și 1.0)
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0

    # Pasul 2: Transformarea semnalului din valori digitale în Pascali [Pa]
    # p(t) = semnal_digital(t) * constanta_calibrare
    presiune_pa = data * c_calib

    # Pasul 3: Definirea referinței absolute din curs (Formula 1.22)
    P_REF = 2e-5  # 20 microPascali

    # Pasul 4: Integrarea temporală pe ferestre (Formula 1.8 și Paragraful 2.1)
    # Fast = 125ms, Slow = 1000ms
    durata_fereastra = 0.125 if mod_timp == "Fast" else 1.0
    dimensiune_fereastra = int(rate * durata_fereastra)
    total_esantioane = len(presiune_pa)
    timp_axe = []
    spl_stanga = []
    spl_dreapta = []

    # Parcurgem semnalul din aproape în aproape (Integrare pe intervalul T)
    for i in range(0, total_esantioane, dimensiune_fereastra):
        fereastra = presiune_pa[i : i + dimensiune_fereastra]

        if len(fereastra) < dimensiune_fereastra:
            break

        # Separăm canalele (Stânga / Dreapta)
        p_stanga = fereastra[:, 0]
        p_dreapta = fereastra[:, 1]

        # Calculăm presiunea sonoră efectivă RMS pentru fiecare canal (Formula 1.8)
        p_rms_stanga = np.sqrt(np.mean(p_stanga**2))
        p_rms_dreapta = np.sqrt(np.mean(p_dreapta**2))

        # Pasul 5: Blocul de logaritmare în dB SPL absolute (Formula 1.23)
        # np.clip previne logaritmul din 0 în caz de liniște totală
        db_spl_L = 20 * np.log10(np.clip(p_rms_stanga / P_REF, 1e-5, None))
        db_spl_R = 20 * np.log10(np.clip(p_rms_dreapta / P_REF, 1e-5, None))

        # Salvare timp (mijlocul ferestrei curente)
        moment_timp = (i + dimensiune_fereastra / 2) / rate
        timp_axe.append(moment_timp)
        spl_stanga.append(db_spl_L)
        spl_dreapta.append(db_spl_R)

    return np.array(timp_axe), np.array(spl_stanga), np.array(spl_dreapta)


# --- RULARE SIMULARE ȘI GRAFIC ---
if __name__ == "__main__":
    nume_fisier = "test.wav"  # Fișierul tău stereo

    try:
        # Rulăm sonometrul virtual pe modul Fast
        # c_calib=20.0 înseamnă că un semnal digital maxim (1.0) va fi egal cu 20 Pa în aer.
        timp, spl_L, spl_R = simulator_sonometru_absolut(
            nume_fisier, mod_timp="Fast", c_calib=20.0
        )

        # Generare Grafic în dB SPL absolut
        plt.figure(figsize=(12, 5))
        plt.plot(timp, spl_L, label="Canal Stâng (L)", color="royalblue")
        plt.plot(timp, spl_R, label="Canal Drept (R)", color="darkorange")

        plt.title("Nivelul Presiunii Sonore Absolute [dB SPL] - Conform Cap. 1 & 2")
        plt.xlabel("Timp (secunde)")
        plt.ylabel("Nivel Sonor (dB SPL)")

        # Setăm axa Y conform dinamicii auzului descrisă în curs (0 dB auz - 120 dB concert rock)
        plt.ylim(0, 130)
        plt.axhline(
            y=120,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Prag disconfort (120 dB)",
        )

        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.show()

    except FileNotFoundError:
        print(f"Eroare: Nu s-a găsit fișierul '{nume_fisier}'.")