# BERTax Cacao
Esta herramienta sirve para la limpieza de secuencias crudas de Theobroma cacao.

Usar modelo construído en (/bertax_cacao_training)[https://github.com/mariana2323/bertax_cacao_training] o el disponible en (fine-tune-BERTaxCacao.keras)[https://drive.google.com/drive/folders/1-DQ2VUiPc4cQ0HmFIAZ5oKY_OYGiTg6W?usp=sharing]

Esta herramienta necesita un archivo fasta de entrada.

Se recomienda ejecutar el notebook en un runtime con GPU en Google colab o un contenedor que disponga de GPU.

Para más de 2 millones de secuencias, la herramienta tardó 7 minutos para un runtime type A100 de Google Colab.

El output de esta herramienta es un archivo confidencias.json.

Los siguientes scripts pueden ser ejecutados para visualizar los resultados de la predicción para la limpieza de secuencias crudas de Theobroma cacao:

