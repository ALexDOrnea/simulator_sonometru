import sys
import numpy as np
from scipy.interpolate import PchipInterpolator

F_MIN,F_MAX=20.0,20000.0
N_PUNCTE=300          # ~100 puncte/decada, ca in 35Y228_cal_0degree.txt


def read_csv(path):
    f,a,b=[],[],[]
    with open(path,encoding="utf-8-sig") as file:
        for nr,line in enumerate(file,1):
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line:
                fields=[x.replace(",",".") for x in line.split(";")]
            else:
                fields=line.replace(","," ").split()
            try:
                f_i,a_i,b_i=(float(x) for x in fields[:3])
            except ValueError:
                if nr==1:
                    continue
                print(f"Line {nr} ignored")
                continue
            f.append(f_i)
            a.append(a_i)
            b.append(b_i)
    return np.array(f),np.array(a),np.array(b)


def main():
    csv_path=input("path to csv file> ").strip().strip("'\"")
    try:
        f,val_script,val_ref=read_csv(csv_path)
    except OSError as e:
        print(f"Cannot read file {e}")
        sys.exit(1)

    error=val_script-val_ref

    # sortare si mediere pentru frecvente duplicate
    order=np.argsort(f)
    f,error=f[order],error[order]
    f_unic,inv=np.unique(f,return_inverse=True)
    eroare_unica=np.bincount(inv,weights=error)/np.bincount(inv)

    if len(f_unic)<2:
        print("You need at least 2 different frequencies")
        sys.exit(1)
    if np.any(f_unic<=0):
        print("Freq needs to be positive")
        sys.exit(1)

    print(f"\n{len(f_unic)} values read: {f_unic[0]:g}Hz -- {f_unic[-1]:g}Hz")
    for fi, ei in zip(f_unic,eroare_unica):
        print(f"{fi:10.1f}Hz error={ei:+.2f}dB")

    # interpolare PCHIP pe axa logaritmica (fara overshoot)
    interp=PchipInterpolator(np.log10(f_unic),eroare_unica)

    # grila logaritmica, frecventele rotunjite la o zecimala ca in fisierul de referinta
    f_out=np.round(np.geomspace(F_MIN,F_MAX, N_PUNCTE),1)
    # in afara intervalului masurat eroarea ramane constanta
    x=np.clip(np.log10(f_out),np.log10(f_unic[0]),np.log10(f_unic[-1]))
    e_out=interp(x)

    name=input("\nInput calibration file name> ").strip()
    if not name:
        name="generated_calibration.txt"
    if "." not in name:
        name+=".txt"

    np.savetxt(name,np.column_stack([f_out,e_out]),fmt="%.1f\t%.2f")
    print(f"Saved: {name} ({len(f_out)} lines, {F_MIN:g}-{F_MAX:g} Hz, logarithmic grid)")
    if f_unic[0]>F_MIN or f_unic[-1]<F_MAX:
        print(f"Attention: outside {f_unic[0]:g}--{f_unic[-1]:g}Hz the error is held constant")

    try:
        import matplotlib.pyplot as plt
        plt.semilogx(f_out, e_out, label="interpolated")
        plt.semilogx(f_unic, eroare_unica, "o", label="measured")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Error[dB]")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()