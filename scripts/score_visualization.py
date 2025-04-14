import json
import matplotlib.pyplot as plt

# Cargar el nuevo JSON (formato: dict con claves como "seq_0")
with open("C:/Users/Ana Acosta/Documents/Tesis/bertax_results/def_1/confidencias.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Inicializar listas
cacao_scores = []
notcacao_scores = []

# Extraer probabilidades según clase predicha
for seq_id, result in data.items():
    if result["Predicted"] == "Cacao":
        cacao_scores.append(result["Cacao"])
    else:
        notcacao_scores.append(result["NotCacao"])

# Graficar histogramas
plt.hist(cacao_scores, bins=30, alpha=0.6, label="Cacao", color='brown')
plt.hist(notcacao_scores, bins=30, alpha=0.6, label="No Cacao", color='green')
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia")
plt.title("Distribución de scores por clase")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
