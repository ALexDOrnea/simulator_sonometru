import warnings
import os
import sys
import queue
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import bilinear, lfilter, lfilter_zi, butter

# ignore wav warnings
warnings.filterwarnings("ignore",category=UserWarning,module="scipy.io.wavfile")


################################################
############Select in/out devices###############
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE=True
except ImportError:
    HAS_SOUNDDEVICE=False
    print("Module 'sounddevice' not found.Audio playback will be disabled.")
    sys.exit()

input_device_id=None
output_device_id=None

if HAS_SOUNDDEVICE:
    print("\n~~~~~CONFIGURARE DISPOZITIVE AUDIO~~~~~")
    devices=sd.query_devices()
    
    # 1. select input
    print("\nselect input:")
    input_indices=[]
    for i,dev in enumerate(devices):
        if dev['max_input_channels']>0:
            print(f"[{i}] {dev['name']} (Canale in:{dev['max_input_channels']})")
            input_indices.append(i)
            
    try:
        id_in=input("select input device (sau Enter pentru implicit): ")
        if id_in.strip()!="":
            id_in=int(id_in)
            if id_in in input_indices:
                input_device_id=id_in
            else:
                print("ID invalid. Se va folosi cel implicit.")
    except ValueError:
        print("Introducere invalida. Se va folosi cel implicit.")

    # 2. select output
    print("\nselect output:")
    output_indices=[]
    for i,dev in enumerate(devices):
        if dev['max_output_channels']>0:
            print(f"[{i}] {dev['name']} (Canale out: {dev['max_output_channels']})")
            output_indices.append(i)
            
    try:
        id_out=input("alege output (sau Enter pentru implicit): ")
        if id_out.strip()!="":
            id_out=int(id_out)
            if id_out in output_indices:
                output_device_id = id_out
            else:
                print("ID invalid. Se va folosi cel implicit.")
    except ValueError:
        print("Introducere invalida. Se va folosi cel implicit.")

    # setup global
    sd.default.device=(input_device_id, output_device_id)
    
    # afisare finala
    current_in=sd.query_devices(sd.default.device[0])['name'] if sd.default.device[0] is not None else "default"
    current_out=sd.query_devices(sd.default.device[1])['name'] if sd.default.device[1] is not None else "default"
    print(f"\nConfiguratie activata:\n   -> Input: {current_in}\n   -> Output: {current_out}\n")


########################################################
################# SELECTIE SURSA #######################
print("~~~~~CONFIGURARE SURSA SEMNAL~~~~~")
print("1.fisier WAV")
print("2.microfon")
SURSA_OPT=input("Alege sursa: ").strip()

WAV_PATH=input("insert file path: ")
    
MODE=input("Insert mode (Fast,Slow,Peak): ").strip()


print("\n~~~~~SELECTIE GRAFICE VIZUALE~~~~~")
print("1 grafic dB FS")
print("2 grafic FFT")
print("3 toate")
GRAFIC_OPT=input("alege tipul de afisare: ").strip()
if GRAFIC_OPT not in ["1","2","3"]:
    print("optiune invalida se alege 1 automat(default)")
    GRAFIC_OPT="1"

#setari initiale
SAMPLE_RATE=44100
EPSILON=1e-12
AUDIO_NORM=None

# Coada thread-safe prin care trimitem datele de la sunet către grafic
# Acum trimitem un tuplu format din: (timp_curent, valoare_db, spectru_fft)
data_queue=queue.Queue()
play_pointer=0

# Pregătire date în funcție de sursă
if SURSA_OPT=="1":
    if not os.path.exists(WAV_PATH):
        print(f"Fisierul {WAV_PATH} nu a fost gasit! Verifica calea.")
        sys.exit()

    SAMPLE_RATE, AUDIO_DATA=wavfile.read(WAV_PATH)

    if len(AUDIO_DATA.shape)>1:
        AUDIO_DATA=AUDIO_DATA[:,0]
        print("Audio transformat mono")
    
    if AUDIO_DATA.dtype==np.int16:
        AUDIO_NORM=AUDIO_DATA/32768.0
    elif AUDIO_DATA.dtype==np.int32:
        AUDIO_NORM=AUDIO_DATA/2147483648.0
    else:
        AUDIO_NORM=AUDIO_DATA

# ========================================================
# PROIECTAREA FILTRULUI HIGH-PASS
# ========================================================
# IMPORTANT: se face DUPA citirea WAV-ului, pentru ca SAMPLE_RATE
# se poate schimba (wavfile.read suprascrie SAMPLE_RATE cu valoarea
# reala din fisier). Daca proiectam filtrul cu SAMPLE_RATE-ul vechi
# (44100 implicit), cutoff_norm ar fi calculat gresit pentru un WAV
# cu alta rata de esantionare (ex: 48000).
CUTOFF_HZ = 1000       # schimba cu frecventa de taiere dorita
ORDIN_FILTRU = 4

nyquist = SAMPLE_RATE / 2
cutoff_norm = CUTOFF_HZ / nyquist
b_hp, a_hp = butter(ORDIN_FILTRU, cutoff_norm, btype='high')
zi_hp = lfilter_zi(b_hp, a_hp) * 0.0   # stare initiala, tacere

def filtreaza_block(chunk):
    """Aplica filtrul high-pass pe un bloc de audio, pastrand starea intre apeluri."""
    global zi_hp
    chunk_filtrat, zi_hp = lfilter(b_hp, a_hp, chunk, zi=zi_hp)
    return chunk_filtrat

# Liste in care acumulam tot semnalul (brut si filtrat), bloc cu bloc,
# ca sa putem construi la final graficul de comparatie spectrala completa.
semnal_nefiltrat_complet = []
semnal_filtrat_complet = []

# Configurare WINDOW_SIZE în funcție de mod
if MODE.lower()=="fast":
    WINDOW_SIZE=int(0.125*SAMPLE_RATE)
elif MODE.lower()=="slow":
    WINDOW_SIZE=int(1.0*SAMPLE_RATE)
elif MODE.lower()=="peak":
    WINDOW_SIZE=int(0.035*SAMPLE_RATE)
else: 
    print("Mod inexistent")
    sys.exit()

# Buffer-ul circular în care calculăm RMS-ul și FFT-ul
live_ring_buffer=np.zeros(WINDOW_SIZE)

# Frecvențele corespunzătoare pentru FFT
fft_frequencies=np.fft.rfftfreq(WINDOW_SIZE,d=1.0/SAMPLE_RATE)


# ========================================================
# PROCESARE AUDIO & ANALIZA SPECTRALĂ (CALLBACKS)
# ========================================================

def proceseaza_block_audio(chunk,frames):
    """Calculează nivelul dB și spectrul FFT pentru buffer-ul curent"""
    global live_ring_buffer
    
    # Actualizăm ring-buffer-ul
    live_ring_buffer=np.roll(live_ring_buffer,-frames)
    live_ring_buffer[-frames:]=chunk
    
    # 1. Calcul dB (RMS)
    rms=np.sqrt(np.mean(np.square(live_ring_buffer)))
    db=20 *np.log10(rms+EPSILON)
    db=np.clip(db,-120.0, 0.0)
    
    # 2. Calcul FFT (aplicăm și o fereastră Hanning ca să reducem zgomotul spectral de margine)
    hanning_window=np.hanning(WINDOW_SIZE)
    windowed_signal=live_ring_buffer*hanning_window
    
    fft_raw = np.abs(np.fft.rfft(windowed_signal))
    # Normalizăm FFT-ul și îl convertim în dB
    fft_norm = fft_raw / (WINDOW_SIZE / 2)
    fft_db = 20 * np.log10(fft_norm + EPSILON)
    fft_db = np.clip(fft_db, -120.0, 0.0)
    
    return db, fft_db


# Callback pentru redare WAV (Modul 1)
def playback_callback(outdata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
    
    chunk = AUDIO_NORM[play_pointer : play_pointer + frames]
    
    if len(chunk) < frames:
        # 1. FILTREZ
        chunk_filtrat = filtreaza_block(chunk)
        outdata[:len(chunk), 0] = chunk_filtrat
        outdata[len(chunk):, 0] = 0
        semnal_nefiltrat_complet.append(chunk.copy())
        semnal_filtrat_complet.append(chunk_filtrat.copy())
        raise sd.CallbackStop()
    else:
        # 1. FILTREZ
        chunk_filtrat = filtreaza_block(chunk)
        outdata[:, 0] = chunk_filtrat
        play_pointer += frames
        semnal_nefiltrat_complet.append(chunk.copy())
        semnal_filtrat_complet.append(chunk_filtrat.copy())

    # 2. ANALIZEZ (pe semnalul deja filtrat)
    db, fft_db = proceseaza_block_audio(chunk_filtrat, frames)
    
    # 3. trimit spre PLOTARE (se intampla in bucla principala)
    current_time = play_pointer / SAMPLE_RATE
    data_queue.put((current_time, db, fft_db))


# Callback pentru microfon (Modul 2)
def record_callback(indata, frames, time_info, status):
    global play_pointer
    if status:
        print(status, file=sys.stderr)
        
    chunk = indata[:, 0]
    play_pointer += frames

    # 1. FILTREZ
    chunk_filtrat = filtreaza_block(chunk)
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_filtrat.copy())

    # 2. ANALIZEZ (pe semnalul deja filtrat)
    db, fft_db = proceseaza_block_audio(chunk_filtrat, frames)
    
    # 3. trimit spre PLOTARE
    current_time = play_pointer / SAMPLE_RATE
    data_queue.put((current_time, db, fft_db))


# ========================================================
# CONSTRUIREA INTERFEȚEI GRAFICE ACTIVE (MATPLOTLIB)
# ========================================================

plt.ion()  # Modul interactiv pornit

# Inițializăm axele în funcție de opțiunea aleasă
if GRAFIC_OPT == "1":
    fig, ax_db = plt.subplots(figsize=(10, 4))
    line_db, = ax_db.plot([], [], color="purple", label=f"Nivel rolling ({MODE})")
    ax_fft = None
elif GRAFIC_OPT == "2":
    fig, ax_fft = plt.subplots(figsize=(10, 4))
    line_fft, = ax_fft.plot(fft_frequencies, np.zeros_like(fft_frequencies), color="cyan")
    ax_db = None
elif GRAFIC_OPT == "3":
    # Două subplot-uri separate (sus dB, jos FFT)
    fig, (ax_db, ax_fft) = plt.subplots(2, 1, figsize=(10, 8))
    line_db, = ax_db.plot([], [], color="purple", label=f"Nivel rolling ({MODE})")
    line_fft, = ax_fft.plot(fft_frequencies, np.zeros_like(fft_frequencies), color="cyan")

# Formatare axă dB
if ax_db is not None:
    ax_db.set_ylim(-120, 0)
    ax_db.set_xlabel("Timp (s)")
    ax_db.set_ylabel("Nivel (dB FS)")
    ax_db.grid(True)
    ax_db.legend()
    if SURSA_OPT == "1":
        duration = len(AUDIO_NORM) / SAMPLE_RATE
        ax_db.set_xlim(0, duration)
        ax_db.set_title(f"Analiza sonometru (Fisier WAV, filtrat) - Mod: {MODE}")
    else:
        ax_db.set_xlim(0, 10)
        ax_db.set_title(f"Analiza sonometru LIVE (Microfon, filtrat) - Mod: {MODE}")

# Formatare axă FFT
if ax_fft is not None:
    ax_fft.set_xlim(20, 20000)  # Limite de frecvență audibile (20 Hz - 20 kHz)
    ax_fft.set_xscale('log')    # Scara logaritmică este standard în audio
    ax_fft.set_ylim(-100, 0)
    ax_fft.set_xlabel("Frecventa (Hz)")
    ax_fft.set_ylabel("Amplitudine (dB FS)")
    ax_fft.grid(True, which="both")
    ax_fft.set_title("Analiza spectrala FFT (Timp Real, filtrat)")

# Liste pentru stocarea istoricului de dB
x_data = []
y_data = []

# Pornirea stream-ului audio
try:
    if SURSA_OPT == "1":
        print("🔊 Se reda audio si se randeaza graficele live...")
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=playback_callback)
    elif SURSA_OPT == "2":
        print("🎤 Se preia semnal live si se randeaza graficele...")
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=record_callback)
    else:
        print("Optiune invalida.")
        sys.exit()

    with stream:
        while stream.active and plt.fignum_exists(fig.number):
            updated = False
            last_fft_data = None
            
            # Citim toate pachetele disponibile din coadă
            while True:
                try:
                    t, db, fft_db = data_queue.get_nowait()
                    x_data.append(t)
                    y_data.append(db)
                    last_fft_data = fft_db
                    updated = True
                    
                    # Printăm valoarea brută în consolă pentru monitorizare exactă
                    print(f"Timp: {t:6.2f}s | Nivel: {db:6.1f} dBFS")
                except queue.Empty:
                    break
            
            if updated:
                # 1. Updatează graficul de dB
                if ax_db is not None:
                    if SURSA_OPT == "2" and x_data[-1] > 10:
                        ax_db.set_xlim(x_data[-1] - 10, x_data[-1])
                    line_db.set_data(x_data, y_data)
                
                # 2. Updatează graficul spectral (FFT) cu ultimul spectru calculat
                if ax_fft is not None and last_fft_data is not None:
                    line_fft.set_ydata(last_fft_data)
                
                # Redesenăm frame-ul pe ecran
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            plt.pause(0.01)

except KeyboardInterrupt:
    print("\n=Monitorizare oprita")

finally:
    plt.ioff()
    if HAS_SOUNDDEVICE:
        sd.stop()
    print("\nProcesare finalizata")

    # ========================================================
    # GRAFIC FINAL: comparatie spectru NEFILTRAT vs FILTRAT
    # (calculat pe TOT semnalul acumulat, nu doar pe fereastra live)
    # ========================================================
    if len(semnal_nefiltrat_complet) > 0:
        semnal_complet_raw  = np.concatenate(semnal_nefiltrat_complet)
        semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

        N = len(semnal_complet_raw)
        freqs_finale = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)

        fft_raw_orig = np.abs(np.fft.rfft(semnal_complet_raw))  / (N / 2)
        fft_raw_filt = np.abs(np.fft.rfft(semnal_complet_filt)) / (N / 2)

        db_orig = np.clip(20 * np.log10(fft_raw_orig + EPSILON), -120, 0)
        db_filt = np.clip(20 * np.log10(fft_raw_filt + EPSILON), -120, 0)

        fig2, ax_comp = plt.subplots(figsize=(10, 5))

        ax_comp.plot(freqs_finale, db_orig, color="orange", label="Spectru nefiltrat (original)")
        ax_comp.plot(freqs_finale, db_filt, color="cyan",   label="Spectru filtrat (high-pass)")
        ax_comp.axvline(CUTOFF_HZ, color="gray", linestyle="--", linewidth=1,
                         label=f"Cutoff ({CUTOFF_HZ} Hz)")

        ax_comp.set_xscale('log')
        ax_comp.set_xlim(20, 20000)
        ax_comp.set_ylim(-120, 0)
        ax_comp.set_title(f"Comparatie spectrala: original vs high-pass @ {CUTOFF_HZ} Hz (ordin {ORDIN_FILTRU})")
        ax_comp.set_xlabel("Frecventa (Hz)")
        ax_comp.set_ylabel("Amplitudine (dB FS)")
        ax_comp.grid(True, which="both")
        ax_comp.legend()

        fig2.tight_layout()

    plt.show()