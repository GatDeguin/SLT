# Herramientas

## Dependencias

Las utilidades ubicadas en este directorio requieren Python 3.9+ y las
siguientes bibliotecas adicionales:

```bash
pip install opencv-python mediapipe
```

`opencv-python` se usa para lectura/escritura de imágenes y videos, mientras
que `mediapipe` proporciona los detectores de cara, manos y pose necesarios
para `extract_rois_v2.py`.
