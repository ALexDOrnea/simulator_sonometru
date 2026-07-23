import warnings
import os
import sys
import queue
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter,lfilter_zi,butter,sosfilt,sosfilt_zi,bilinear_zpk,zpk2sos

## pentru ignorare avertismente wav
warnings.filterwarnings("ignore",category=UserWarning,module="scipy.io.wavfile")

## logica selectare dispozitive pentru input/output

###########################################
############Selectie in/out################

# deschidere sounddevice
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE=True
except ImportError:
    HAS_SOUNDDEVICE=False
    print("Sounddevice nu a fost initializat. incearca sa il instalezi")
    sys.exit()

# pornim cu cazul default
input_device_id=None
output_device_id=None

if HAS_SOUNDDEVICE:
    print("\n\n~~~~~CONFIG AUDIO~~~~~")
    devices=sd.query_devices()

    # print input devices
    print("\nOptiuni input")
    input_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_input_channels"]>0:
            print(f"[{i}] {dev['name']} (Input channels: {dev['max_input_channels']})")
            input_indices.append(i)

    # select input
    try:
        id_in=input("select device(blank for default)\n> ").strip()
        if id_in:
            id_in=int(id_in)
            if id_in in input_indices:
                input_device_id=id_in
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    # print output devices
    print("\nOptiuni output")
    output_indices=[]
    for i,dev in enumerate(devices):
        if dev["max_output_channels"]>0:
            print(f"[{i}] {dev['name']} (Output channels: {dev['max_output_channels']})")
            output_indices.append(i)

    # select output
    try:
        id_out=input("select device(blank for default)\n> ").strip()
        if id_out:
            id_out=int(id_out)
            if id_out in output_indices:
                output_device_id=id_out
            else:
                print("Invalid ID using default")
    except ValueError:
        print("invalid value.using default")

    # logica default
    sd.default.device=(input_device_id,output_device_id)
    current_in=(
        sd.query_devices(sd.default.device[0])["name"]
        if sd.default.device[0] is not None
        else "default"
    )
    current_out=(
        sd.query_devices(sd.default.device[1])["name"]
        if sd.default.device[1] is not None
        else "default"
    )
    # print final
    print(f"\n\nConfiguration:\nInput={current_in}\nOutput={current_out}\n")

###########################################
############Alte setari####################

print("~~~~~CONFIGURARE SURSA SEMNAL~~~~~")
print("1. WAV")
print("2. Live")
SURSA_OPT=input("> ").strip()

WAV_PATH=None
if SURSA_OPT=="1":
    print("~~~~~CONFIGURARE PATH WAV~~~~~")
    WAV_PATH=input("path to wav file: ").strip().strip('"')
elif SURSA_OPT!="2":
    print("Invalid option")
    sys.exit()

print("~~~~~CONFIGURARE MOD~~~~~")
MODE=input("Mode(Fast, Slow, Peak)\n>  ").strip()

print("\n~~~~~SELECTIE FILTRU~~~~~")
print("1 Low-pass")
print("2 High-pass")
print("3 A-Weighting (IEC 61672)")
print("4 C-Weighting (IEC 61672)")
FILTRU_OPT=input("> ").strip().lower()

if FILTRU_OPT in ("1","lowpass","low-pass","low","lp"):
    TIP_FILTRU="lowpass"
    BTYPE_SCIPY="lowpass"
elif FILTRU_OPT in ("2","highpass","high-pass","high","hp"):
    TIP_FILTRU="highpass"
    BTYPE_SCIPY="highpass"
elif FILTRU_OPT in ("3", "a", "a-weighting", "a-weight"):
    TIP_FILTRU = "A-Weighting"
elif FILTRU_OPT in ("4", "c", "c-weighting", "c-weight"):
    TIP_FILTRU = "C-Weighting"
else:
    print("Invalid option.using highpass")
    TIP_FILTRU="highpass"
    BTYPE_SCIPY="highpass"

if TIP_FILTRU=="highpass" or TIP_FILTRU=="lowpass":
    print("\n~~~~~SELECTIE FRECV TAIERE~~~~~")
    try:
        CUTOFF_HZ=float(input("Cutoff freq\n> ").strip())
    except ValueError:
        print("invalid freq")
        sys.exit()

    print("\n~~~~~SELECTIE ORDIN FILTRU~~~~~")
    try:
        ORDIN_FILTRU=float(input("filter order\n> ").strip())
    except ValueError:
        print("invalid value,using 4")
        ORDIN_FILTRU = 4

print("\n~~~~~SELECTIE GRAFICE~~~~~")
print("1 dB FS")
print("2 FFT")
print("3 All")
GRAFIC_OPT=input("> ").strip()
if GRAFIC_OPT not in ("1","2","3"):
    print("Invalid option.using all")
    GRAFIC_OPT = "3"

# print(ORDIN_FILTRU)

###########################################
#########Setari initiale DSP###############
SAMPLE_RATE=44100
EPSILON=1e-12
AUDIO_NORM=None
#pentru rolling buffer
data_queue=queue.Queue()
play_pointer=0


###########################################
#########CITIRE WAV SI PREGATIRE SEMNAL####
#citim fisierul inainte de proiectarea filtrului, deoarece rata de
#esantionare reala poate fi diferita de valoarea implicita de 44100 Hz
if SURSA_OPT=="1":
    if not os.path.exists(WAV_PATH):
        print(f"{WAV_PATH} not found")
        sys.exit()

    SAMPLE_RATE,AUDIO_DATA=wavfile.read(WAV_PATH)
    #stereo to mono
    if AUDIO_DATA.ndim>1:
        AUDIO_DATA=AUDIO_DATA[:,0]
        print("Audio transformed to mono")
    #selectare valoare pentru normare
    if AUDIO_DATA.dtype==np.int16:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/32768.0
    elif AUDIO_DATA.dtype==np.int32:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/2147483648.0
    elif np.issubdtype(AUDIO_DATA.dtype,np.integer):
        info=np.iinfo(AUDIO_DATA.dtype)
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)/max(abs(info.min),info.max)
    else:
        AUDIO_NORM=AUDIO_DATA.astype(np.float64)
#verificare nyquist pentru fft si filtru
nyquist=SAMPLE_RATE/2.0
if not 0<CUTOFF_HZ<nyquist:
    print(f"cutoff freq must be between 0 and {nyquist:.1f} Hz.")
    sys.exit()

cutoff_norm=CUTOFF_HZ/nyquist
b_filter,a_filter=butter(ORDIN_FILTRU,cutoff_norm,btype=BTYPE_SCIPY)
zi_filter=lfilter_zi(b_filter,a_filter)*0.0

#funcite callback care filtreaza un chunk(un buffer)
def filtreaza_block(chunk):
    """filtreaza bloc si pastreaza starea foltrului intre apeluri"""
    global zi_filter
    chunk_filtrat,zi_filter=lfilter(b_filter,a_filter,chunk,zi=zi_filter)
    return chunk_filtrat

#esantionare si cuantizare semnal
semnal_nefiltrat_complet=[]
semnal_filtrat_complet=[]

if MODE.lower()=="fast":
    WINDOW_SIZE=int(0.125*SAMPLE_RATE)
elif MODE.lower()=="slow":
    WINDOW_SIZE=int(1.0*SAMPLE_RATE)
elif MODE.lower()=="peak":
    WINDOW_SIZE=int(0.035*SAMPLE_RATE)
else:
    print("not a mode. using fast")
    WINDOW_SIZE=int(0.125*SAMPLE_RATE)

#2 ring buffere pentru a analiza semnal filtrat si nefiltrat
live_ring_buffer_raw=np.zeros(WINDOW_SIZE)
live_ring_buffer_filtered=np.zeros(WINDOW_SIZE)
hanning_window=np.hanning(WINDOW_SIZE)
fft_frequencies=np.fft.rfftfreq(WINDOW_SIZE,d=1.0/SAMPLE_RATE)


########################################################
#############PROCESARE AUDIO SI ANALIZA#################

##functii 
def get_a_weighting_filter(fs):
    """Generează coeficienții SOS ai filtrului A conform IEC 61672-1 (Anexa E)"""
    f1,f2,f3,f4=20.598997,107.65265,737.86223,12194.217
    A1000=-2.000
    p1=-2*np.pi*f1
    p2=-2*np.pi*f2
    p3=-2*np.pi*f3
    p4=-2*np.pi*f4

    z=[0,0,0,0]  #4 zerouri la 0 Hz
    p=[p1,p1,p2,p3,p4,p4] # 6 poli
    k=(2*np.pi*f4)**2*(10**(A1000/20))

    zeros_d,poles_d,gain_d=bilinear_zpk(z,p,k,fs)
    return zpk2sos(zeros_d,poles_d,gain_d)

def get_c_weighting_filter(fs):
    """Generează coeficienții SOS ai filtrului C conform IEC 61672-1 (Anexa E)"""
    f1,f4=20.598997,12194.217
    C1000=-0.062

    p1=-2*np.pi*f1
    p4=-2*np.pi*f4

    z = [0, 0] # 2 zerouri la 0 Hz
    p = [p1, p1, p4, p4] # 4 poli
    k = (2 * np.pi * f4)**2 * (10**(C1000 / 20))

    zeros_d, poles_d, gain_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(zeros_d, poles_d, gain_d)

def actualizeaza_ring_buffer(buffer,chunk):
    """introduce ultimul bloc in ring buffer inclusiv cand blocul e mai mare"""
    frames=len(chunk)
    if frames>=WINDOW_SIZE:
        buffer[:]=chunk[-WINDOW_SIZE:]
    else:
        buffer[:]=np.roll(buffer,-frames)
        buffer[-frames:]=chunk

def calculeaza_db_fft(buffer):
    """calculeaza nivelul RMS in dBFS si spectrul FFT in dBFS"""
    rms=np.sqrt(np.mean(np.square(buffer)))
    db=np.clip(20*np.log10(rms+EPSILON),-120.0,0.0)

    windowed_signal=buffer*hanning_window
    fft_raw=np.abs(np.fft.rfft(windowed_signal))
    fft_norm=fft_raw/(WINDOW_SIZE/2.0)
    fft_db=np.clip(20*np.log10(fft_norm+EPSILON),-120.0,0.0)
    return db,fft_db


def proceseaza_ambele_semnale(chunk_raw,chunk_filtered):
    """analizeaza in paralel blocul original si blocul filtrat"""
    actualizeaza_ring_buffer(live_ring_buffer_raw,chunk_raw)
    actualizeaza_ring_buffer(live_ring_buffer_filtered,chunk_filtered)

    db_raw,fft_raw=calculeaza_db_fft(live_ring_buffer_raw)
    db_filtered,fft_filtered=calculeaza_db_fft(live_ring_buffer_filtered)

    return db_raw,db_filtered,fft_raw,fft_filtered


def trimite_date_live(chunk,chunk_filtrat):
    """calculeaza si pune in coada valorile pentru ambele semnale"""
    db_raw,db_filtered,fft_raw,fft_filtered=proceseaza_ambele_semnale(
        chunk,chunk_filtrat
    )
    current_time=play_pointer/SAMPLE_RATE
    data_queue.put(
        (current_time,db_raw,db_filtered,fft_raw,fft_filtered)
    )


# callback pentru redarea unui WAV
def playback_callback(outdata,frames,time_info,status):
    global play_pointer
    if status:
        print(status,file=sys.stderr)
    chunk = AUDIO_NORM[play_pointer:play_pointer+frames]
    valid_frames=len(chunk)
    if valid_frames==0:
        outdata.fill(0)
        raise sd.CallbackStop()
    chunk_filtrat=filtreaza_block(chunk)
    outdata.fill(0)
    outdata[:valid_frames,0]=chunk_filtrat
    play_pointer+=valid_frames
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_filtrat.copy())
    trimite_date_live(chunk,chunk_filtrat)
    if valid_frames<frames:
        raise sd.CallbackStop()

#callback pentru microfon
def record_callback(indata,frames,time_info,status):
    global play_pointer
    if status:
        print(status,file=sys.stderr)
    chunk=indata[:,0].astype(np.float64,copy=True)
    chunk_filtrat=filtreaza_block(chunk)
    play_pointer+=len(chunk)
    semnal_nefiltrat_complet.append(chunk.copy())
    semnal_filtrat_complet.append(chunk_filtrat.copy())
    trimite_date_live(chunk, chunk_filtrat)

########################################################
############ Interfata grafica Matplotlib ##############
plt.ion()

line_db_raw=None
line_db_filtered=None
line_fft_raw=None
line_fft_filtered=None

if GRAFIC_OPT=="1":
    fig,ax_db=plt.subplots(figsize=(10,4))
    ax_fft=None
elif GRAFIC_OPT=="2":
    fig,ax_fft=plt.subplots(figsize=(10,4))
    ax_db=None
else:
    fig,(ax_db,ax_fft)=plt.subplots(2,1,figsize=(10,8))

#2 curbe cu culori diferite pe graficul dB
if ax_db is not None:
    line_db_raw,=ax_db.plot(
        [],[],color="orange",label=f"Nefiltrat - nivel rolling ({MODE})"
    )
    line_db_filtered,=ax_db.plot(
        [],[],color="purple",label=f"Filtrat{TIP_FILTRU} - nivel rolling ({MODE})"
    )
    ax_db.set_ylim(-120, 0)
    ax_db.set_xlabel("Timp (s)")
    ax_db.set_ylabel("Nivel (dB FS)")
    ax_db.grid(True)
    ax_db.legend()

    if SURSA_OPT=="1":
        duration=len(AUDIO_NORM)/SAMPLE_RATE
        ax_db.set_xlim(0,duration)
        ax_db.set_title(
            f"Nivel live: nefiltrat vs {TIP_FILTRU} @ {CUTOFF_HZ:g} Hz - Mod: {MODE}"
        )
    else:
        ax_db.set_xlim(0,10)
        ax_db.set_title(
            f"Nivel LIVE microfon: nefiltrat vs {TIP_FILTRU} @ {CUTOFF_HZ:g} Hz - Mod: {MODE}"
        )

# Doua curbe cu culori diferite pe graficul FFT.
if ax_fft is not None:
    line_fft_raw, = ax_fft.plot(
        fft_frequencies,
        np.full_like(fft_frequencies, -120.0),
        color="orange",
        label="FFT nefiltrat",
    )
    line_fft_filtered, = ax_fft.plot(
        fft_frequencies,
        np.full_like(fft_frequencies, -120.0),
        color="cyan",
        label=f"FFT filtrat ({TIP_FILTRU})",
    )
    ax_fft.axvline(
        CUTOFF_HZ,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Taiere: {CUTOFF_HZ:g} Hz",
    )

    max_plot_frequency = min(20000, nyquist)
    min_plot_frequency = min(20, max_plot_frequency / 10)
    ax_fft.set_xlim(min_plot_frequency, max_plot_frequency)
    ax_fft.set_xscale("log")
    ax_fft.set_ylim(-120, 0)
    ax_fft.set_xlabel("Frecventa (Hz)")
    ax_fft.set_ylabel("Amplitudine (dB FS)")
    ax_fft.grid(True, which="both")
    ax_fft.legend()
    ax_fft.set_title(
        f"FFT in timp real: nefiltrat vs {TIP_FILTRU} @ {CUTOFF_HZ:g} Hz"
    )

fig.tight_layout()

x_data=[]
y_db_raw=[]
y_db_filtered=[]

########################################################
#################PORNIREA STREAMULUI####################
try:
    if SURSA_OPT=="1":
        print(
            f"se reda audio filtrat {TIP_FILTRU} la {CUTOFF_HZ:g} Hz "
            "si se afiseaza comparatia live"
        )
        stream=sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=playback_callback,
        )
    else:
        print(
            f"se analizeaza microfonul cu filtru{TIP_FILTRU} la {CUTOFF_HZ:g} Hz "
            "si se afiseaza comparatia live"
        )
        stream=sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=record_callback,
        )

    with stream:
        while stream.active and plt.fignum_exists(fig.number):
            updated=False
            last_fft_raw=None
            last_fft_filtered=None

            while True:
                try:
                    (
                        t,
                        db_raw,
                        db_filtered,
                        fft_raw,
                        fft_filtered,
                    ) = data_queue.get_nowait()

                    x_data.append(t)
                    y_db_raw.append(db_raw)
                    y_db_filtered.append(db_filtered)
                    last_fft_raw=fft_raw
                    last_fft_filtered=fft_filtered
                    updated=True

                    print(
                        f"Timp: {t:6.2f}s | "
                        f"Nefiltrat: {db_raw:6.1f} dBFS | "
                        f"Filtrat: {db_filtered:6.1f} dBFS"
                    )
                except queue.Empty:
                    break

            if updated:
                if ax_db is not None:
                    if SURSA_OPT=="2" and x_data[-1]>10:
                        ax_db.set_xlim(x_data[-1]-10,x_data[-1])

                    line_db_raw.set_data(x_data,y_db_raw)
                    line_db_filtered.set_data(x_data,y_db_filtered)

                if ax_fft is not None and last_fft_raw is not None:
                    line_fft_raw.set_ydata(last_fft_raw)
                    line_fft_filtered.set_ydata(last_fft_filtered)

                fig.canvas.draw()
                fig.canvas.flush_events()

            plt.pause(0.01)

except KeyboardInterrupt:
    print("\nmonitorizare oprita de utilizator.")

finally:
    plt.ioff()
    if HAS_SOUNDDEVICE:
        sd.stop()
    print("\nprocesare finalizata")

    ########################################################
    ######## Comparatie spectrala pe tot semnalul #########
    if semnal_nefiltrat_complet:
        semnal_complet_raw = np.concatenate(semnal_nefiltrat_complet)
        semnal_complet_filt = np.concatenate(semnal_filtrat_complet)

        N=len(semnal_complet_raw)
        freqs_finale = np.fft.rfftfreq(N, d=1.0/SAMPLE_RATE)

        final_window=np.hanning(N)
        fft_orig=np.abs(np.fft.rfft(semnal_complet_raw * final_window))/(N / 2.0)
        fft_filt=np.abs(np.fft.rfft(semnal_complet_filt * final_window))/(N / 2.0)

        db_orig=np.clip(20*np.log10(fft_orig+EPSILON),-120, 0)
        db_filt=np.clip(20*np.log10(fft_filt+EPSILON),-120, 0)

        fig2,ax_comp=plt.subplots(figsize=(10,5))
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
        ax_comp.axvline(
            CUTOFF_HZ,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"taiere ({CUTOFF_HZ:g} Hz)",
        )

        ax_comp.set_xscale("log")
        ax_comp.set_xlim(min(20, min(20000,nyquist)/10),min(20000,nyquist))
        ax_comp.set_ylim(-120,0)
        ax_comp.set_title(
            f"comparatie spectrala: original vs {TIP_FILTRU} "
            f"@ {CUTOFF_HZ:g} Hz (ordin {ORDIN_FILTRU})"
        )
        ax_comp.set_xlabel("Frecventa (Hz)")
        ax_comp.set_ylabel("Amplitudine (dB FS)")
        ax_comp.grid(True, which="both")
        ax_comp.legend()
        fig2.tight_layout()

    plt.show()