# CC219-TP-TF-2024-2--CC92-

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

## Descargar modelo para spaCy
python -m spacy download en_core_web_sm