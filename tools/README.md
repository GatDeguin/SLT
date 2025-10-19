# Herramientas

## Dependencias

Las utilidades ubicadas en este directorio requieren Python 3.9+ y las
siguientes bibliotecas adicionales:

```bash
pip install .[media,export]
```

La dependencia `opencv-python` ahora forma parte de los requisitos base del
paquete y `mediapipe` se declara como extra opcional bajo el grupo
`media`. Instala el extra cuando necesites ejecutar `extract_rois_v2.py` u
otras utilidades que dependen de los detectores de MediaPipe. El extra
`export` instala `onnx`, necesario para generar modelos en formato ONNX con
`export_onnx_encoder_v9.py`.

## Demos y pruebas offline

- `demo_realtime_multistream.py` depende de MediaPipe para la detección de rostro/manos/pose y de una GPU con CUDA para alcanzar FPS en tiempo real (puede ejecutarse en CPU con menor rendimiento). Admite modelos TorchScript y ONNX exportados desde el pipeline de entrenamiento y opcionalmente un tokenizador de HuggingFace para decodificar texto.
- `test_realtime_pipeline.py` reutiliza el mismo pipeline sobre videos pregrabados. Puede generar un video anotado (`--output`) y resulta útil para depurar cambios sin requerir una cámara física.
