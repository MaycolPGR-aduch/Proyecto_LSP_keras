# Proyecto LSP (Lenguaje de Señas Peruano)

Este repositorio contiene el primer prototipo del sistema de reconocimiento de señas del alfabeto LSP (Lenguaje de Señas Peruano) utilizando MediaPipe, OpenCV y TensorFlow.

## Estructura del proyecto

- `data/` - Directorio donde se almacenan las muestras capturadas (secuencias de landmarks).  
- `src/` - Código fuente:  
  - `capture.py` - Captura landmarks de MediaPipe.  
  - `preprocess.py` - Preprocesamiento y generación de datasets.  
  - `model.py` - Definición de la arquitectura del modelo.  
  - `train.py` - Entrenamiento del modelo y guardado en formato `.keras`.  
  - `inference.py` - Inferencia en tiempo real.  
- `notebooks/` - Cuadernos para experimentación.  
- `README.md` - Documentación del proyecto.  
- `requirements.txt` - Lista de dependencias.  
- `venv/` o `venv2/` - Entorno virtual de Python (omitido del repositorio).

## Requisitos

- Python 3.10+  
- Dependencias (instalar con pip):  
  ```bash
  pip install mediapipe opencv-python numpy tensorflow scikit-learn
  ```

## Uso

1. Clonar el repositorio:  
   ```bash
   git clone https://github.com/MaycolPGR-aduch/Proyecto_LSP_keras.git
   cd tu_repositorio
   ```
2. Crear y activar el entorno virtual:  
   ```bash
   py -3.10 -m venv venv  # Windows
   .\venv\Scripts\activate  # PowerShell
   ```
3. Instalar dependencias:  
   ```bash
   pip install -r requirements.txt
   ```
4. Capturar datos:  
   ```bash
   cd src
   python capture.py
   ```
5. Preprocesar datos:  
   ```bash
   python preprocess.py
   ```
6. Entrenar el modelo:  
   ```bash
   python train.py
   ```
7. Ejecutar inferencia:  
   ```bash
   python inference.py
   ```

## Contribuir

- Captura más muestras de señas.  
- Añadir augmentación de datos.  
- Mejorar la arquitectura del modelo.  
- Crear una interfaz gráfica o web.

