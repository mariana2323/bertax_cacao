import json
from Bio import SeqIO

# === PARÁMETROS DE ENTRADA ===
json_path = "C:/Users/Ana Acosta/Documents/Tesis/bertax_results/def_1/confidencias.json"
fasta_path = "C:/Users/Ana Acosta/Documents/GitHub/bertax_cacao_training/models/preprocessing/20cacao.fasta"
output_fasta_path = "C:/Users/Ana Acosta/Documents/Tesis/bertax_results/def_1/solo_cacao.fasta"

# === 1. CARGAR IDs CON PREDICTED = "Cacao" ===
with open(json_path, "r", encoding="utf-8") as f:
    confidencias = json.load(f)

cacao_ids = {seq_id for seq_id, datos in confidencias.items() if datos["Predicted"] == "Cacao"}

print(f" {len(cacao_ids)} secuencias clasificadas como Cacao.")

# === 2. EXTRAER Y GUARDAR SECUENCIAS CORRESPONDIENTES ===
with open(output_fasta_path, "w", encoding="utf-8") as out_handle:
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id in cacao_ids:
            SeqIO.write(record, out_handle, "fasta")

print(f" Se guardaron las secuencias filtradas en: {output_fasta_path}")
