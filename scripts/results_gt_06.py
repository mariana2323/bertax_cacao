import json

# Ruta al archivo confidencias.json
input_file = "C:/Users/Ana Acosta/Documents/Tesis/bertax_results/def_1/confidencias.json"

# Cargar datos
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Inicializar contadores y acumuladores
cacao_count = 0
notcacao_count = 0
cacao_sum = 0.0
notcacao_sum = 0.0

# Reasignar predicciones según nuevo umbral
for entry in data.values():
    if entry["Cacao"] > 0.645:
        cacao_count += 1
        cacao_sum += entry["Cacao"]
    else:
        notcacao_count += 1
        notcacao_sum += entry["NotCacao"]

# Calcular promedios
cacao_avg = cacao_sum / cacao_count if cacao_count else 0.0
notcacao_avg = notcacao_sum / notcacao_count if notcacao_count else 0.0

# Mostrar resultados
print("\n RESULTADOS CON UMBRAL PERSONALIZADO (Cacao > 0.6)\n")
print(f" Total 'Cacao'      : {cacao_count}")
print(f" Total 'NotCacao'   : {notcacao_count}")
print(f" Promedio 'Cacao'  : {cacao_avg:.4f}")
print(f" Promedio 'NotCacao': {notcacao_avg:.4f}")
