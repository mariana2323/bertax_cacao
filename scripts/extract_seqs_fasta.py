from Bio import SeqIO

def extraer_secuencias_fasta(input_fasta: str, output_fasta: str, cantidad: int = 1000):
    """
    Extrae las primeras `cantidad` secuencias de un archivo FASTA y las guarda en otro archivo.
    
    :param input_fasta: Ruta del archivo FASTA original.
    :param output_fasta: Ruta del nuevo archivo FASTA con las secuencias extraídas.
    :param cantidad: Número de secuencias a extraer (por defecto 1000).
    """
    # Leer todas las secuencias del archivo original
    registros = list(SeqIO.parse(input_fasta, "fasta"))

    # Cortar las primeras `cantidad` secuencias
    registros_filtrados = registros[:cantidad]

    # Guardar en nuevo archivo FASTA
    SeqIO.write(registros_filtrados, output_fasta, "fasta")
    print(f"Guardadas {len(registros_filtrados)} secuencias en {output_fasta}")


# Ejemplo de uso
extraer_secuencias_fasta("C:/Users/Ana Acosta/Documents/GitHub/bertax_cacao_training/models/preprocessing/20cacao.fasta", "salida_1000.fasta", cantidad=1000)