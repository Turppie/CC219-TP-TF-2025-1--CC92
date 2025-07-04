# Instalación para Windows con GPU

## Requisitos Previos

1. **Python 3.8-3.11** instalado
2. **NVIDIA GPU** con drivers actualizados
3. **CUDA 11.8 o 12.1** instalado (recomendado 11.8 para mayor compatibilidad)

## Opción 1: Instalación Automática (Recomendada)

### Usando Command Prompt:
```bash
install_windows_gpu.bat
```

### Usando PowerShell:
```powershell
.\install_windows_gpu.ps1
```

## Opción 2: Instalación Manual

1. **Actualizar pip:**
```bash
python -m pip install --upgrade pip
```

2. **Instalar PyTorch con CUDA:**
```bash
# Para CUDA 11.8 (más compatible)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# O para CUDA 12.1 (si tienes drivers más nuevos)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. **Instalar dependencias restantes:**
```bash
pip install -r requirements.txt
```

4. **Descargar modelo de spaCy:**
```bash
python -m spacy download en_core_web_sm
```

5. **Descargar datos de NLTK:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```

## Verificación de Instalación

Ejecuta este código para verificar que GPU esté disponible:

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
```

## Solución de Problemas

### Error: "CUDA not available"
- Verifica que tengas drivers NVIDIA actualizados
- Verifica que CUDA esté instalado correctamente
- Reinstala PyTorch con el comando CUDA correcto

### Error de memoria GPU
- Reduce `per_device_train_batch_size` de 16 a 8 o 4
- Reduce `max_length` de 128 a 64

### Error de compatibilidad de versiones
- Usa un entorno virtual: `python -m venv bert_env`
- Activa el entorno: `bert_env\Scripts\activate`
- Instala las dependencias en el entorno virtual

## Rendimiento Esperado

Con GPU NVIDIA (RTX 3060 o superior):
- Entrenamiento: ~10-15 minutos
- Inferencia: ~0.1 segundos por tweet

Con CPU:
- Entrenamiento: ~45-90 minutos
- Inferencia: ~0.5 segundos por tweet

## Archivos Importantes

- `requirements.txt`: Todas las dependencias
- `requirements-gpu.txt`: Solo PyTorch con CUDA
- `install_windows_gpu.bat`: Script de instalación para CMD
- `install_windows_gpu.ps1`: Script de instalación para PowerShell
