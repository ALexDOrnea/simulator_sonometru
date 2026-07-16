import matplotlib.pyplot as plt

# --- VALORILE TALE DE TEST (Pune aici vectorul tău exact) ---
VALORI_TEST = [-5.17, -4.37, -3.94, -4.09, -6.69, 2.00, 6.12, 0.89, -22.14, -19.00, -11.15, -10.21, -52.75, -51.56]

# Generăm indecșii pe axa X (0, 1, 2, ...) corespunzători fiecărei valori
indecsi = list(range(len(VALORI_TEST)))

# --- GRAFIC ---
plt.figure(figsize=(10, 4))

plt.plot(
    indecsi, 
    VALORI_TEST, 
    color="purple", 
    marker="o", 
    linestyle="-", 
    linewidth=2, 
    label="Valori din vector"
)

plt.title("Plotare Valori de Test")
plt.xlabel("Index element în vector")
plt.ylabel("Valoare")
plt.grid(True)
plt.legend()

plt.show()