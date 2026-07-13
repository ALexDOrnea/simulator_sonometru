import math
import struct
import wave

# Configurări fișier audio
filename = "complex_stereo_melody.wav"
duration = 12.0  # Durata totală în secunde
sample_rate = 44100  # 44.1 kHz
num_channels = 2  # Stereo
sample_width = 2  # 16-bit
max_amplitude = 32767

# Definim o progresie de frecvențe (note muzicale în Hz)
# Canal Stânga: va cânta note mai joase (fundamentul/basul)
melody_left = [130.81, 146.83, 164.81, 196.00, 164.81, 146.83]  # Do2, Re2, Mi2, Sol2...
# Canal Dreapta: va cânta note mai înalte (arpegiu/armonie)
melody_right = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63]  # Do3, Mi3, Sol3, Do4...

# Cât timp stă o notă înainte să treacă la următoarea (în secunde)
note_duration = 0.5  

total_samples = int(sample_rate * duration)

print(f"Se generează melodia stereo complexă în '{filename}'...")

with wave.open(filename, "w") as wav_file:
    wav_file.setparams((num_channels, sample_width, sample_rate, total_samples, "NONE", "not compressed"))
    
    audio_frames = bytearray()
    
    for i in range(total_samples):
        t = i / sample_rate
        
        # 1. Calculăm ce notă trebuie să cânte în funcție de timp
        current_note_index = int(t / note_duration)
        
        # Folosim modulo (%) ca să reluăm lista de note de la capăt când se termină
        freq_l = melody_left[current_note_index % len(melody_left)]
        freq_r = melody_right[current_note_index % len(melody_right)]
        
        # 2. Modularea volumului (creăm un efect de puls/tremolo diferit pe fiecare canal)
        # Canalul stânga pulsează încet, cel din dreapta pulsează mai repede
        vol_modifier_l = 0.3 + 0.3 * math.sin(2 * math.pi * 0.5 * t)  # Volumul variază între 0% și 60%
        vol_modifier_r = 0.5 + 0.4 * math.cos(2 * math.pi * 1.5 * t)  # Volumul variază între 10% și 90%
        
        # 3. Atenuare la începutul și sfârșitul fiecărei note (ca să nu sune ca un click artificial)
        time_within_note = t % note_duration
        fade = 1.0
        if time_within_note < 0.05:  # Fade in de 50ms la începutul notei
            fade = time_within_note / 0.05
        elif time_within_note > (note_duration - 0.05):  # Fade out de 50ms la final
            fade = (note_duration - time_within_note) / 0.05
            
        # 4. Generarea undelor (combinăm o undă sinusoidală cu o undă triunghiulară subtilă pentru un sunet mai bogat)
        # Canal Stânga
        sine_l = math.sin(2 * math.pi * freq_l * t)
        triangle_l = 2 * abs(2 * (t * freq_l - math.floor(t * freq_l + 0.5))) - 1
        wave_l = (sine_l * 0.7) + (triangle_l * 0.3)  # mix de texturi
        val_left = int(wave_l * max_amplitude * vol_modifier_l * fade)
        
        # Canal Dreapta
        sine_r = math.sin(2 * math.pi * freq_r * t)
        triangle_r = 2 * abs(2 * (t * freq_r - math.floor(t * freq_r + 0.5))) - 1
        wave_r = (sine_r * 0.7) + (triangle_r * 0.3)
        val_right = int(wave_r * max_amplitude * vol_modifier_r * fade)
        
        # Împachetarea stereo
        packed_left = struct.pack('<h', val_left)
        packed_right = struct.pack('<h', val_right)
        
        audio_frames.extend(packed_left)
        audio_frames.extend(packed_right)
        
    wav_file.writeframes(audio_frames)

print("Gata! Fișierul complex a fost salvat.")