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

- `demo_realtime_multistream.py` usa MediaPipe para detectar rostro/manos/pose y se beneficia de una
  GPU CUDA. Puede correr en CPU con menor rendimiento. Requiere un modelo (`--model`) o
  declarar `--model-format stub`, junto a un tokenizador de HuggingFace (`--tokenizer`).
- `test_realtime_pipeline.py` reutiliza el pipeline con videos pregrabados y puede generar un video
  anotado (`--output`). Resulta útil para depurar cambios sin requerir una cámara física.
- `eval_slt_multistream_v9.py` valida el tokenizador con `slt.utils.validate_tokenizer` antes de
  procesar los lotes y reporta BLEU, ChrF, CER y WER empleando las utilidades de
  `slt.utils.text`. Reutiliza `slt.utils.decode` para reconstruir las secuencias generadas.
