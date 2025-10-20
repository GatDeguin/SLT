# SLT — Demo de entrenamiento

Este repositorio reúne los componentes mínimos para entrenar y evaluar el stub
multi-stream sobre el conjunto `single_signer`. El paquete expone una CLI
(`python -m slt`) que arma el dataset a partir de `meta.csv`, ejecuta un ciclo
de entrenamiento breve y guarda los checkpoints resultantes. La documentación
en `docs/` amplía cada etapa y el archivo [AGENTS.md](AGENTS.md) resume las
normas de contribución.

## Instalación

Configura un entorno virtual y utiliza el archivo de requisitos para instalar
las dependencias obligatorias y opcionales.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows utiliza `.venv\Scripts\activate`
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` instala el paquete en modo editable y habilita los extras
registrados en `pyproject.toml`:

| Extra | Propósito |
|-------|-----------|
| `media` | Seguimiento facial y de manos con MediaPipe para extracción de ROI. |
| `metrics` | Cálculo de métricas (BLEU, CHRF, CER) durante la evaluación. |
| `export` | Exportación del encoder a ONNX y TorchScript. |

Si prefieres instalarlos de forma explícita puedes ejecutar
`pip install .[media,metrics,export]` tras la instalación base.

## Flujo completo con `single_signer`

Los ejemplos utilizan el directorio `data/single_signer/` como destino para los
artefactos intermedios. Asegúrate de que `meta.csv` (incluido en el repositorio)
se encuentre en la raíz del proyecto o ajusta las rutas según tu estructura.

1. **Organiza los vídeos fuente** en `data/single_signer/videos/`.
2. **Extrae las regiones de interés** con MediaPipe para generar los streams
   sincronizados:

   ```bash
   python tools/extract_rois_v2.py \
     --videos data/single_signer/videos \
     --output data/single_signer/processed \
     --metadata meta.csv
   ```

   Este paso crea las carpetas `face/`, `hand_l/`, `hand_r/` y `pose/` dentro de
   `data/single_signer/processed/` y normaliza los recortes.
3. **Construye los splits** a partir de `meta.csv`. Un ejemplo rápido para
   generar particiones pequeñas es:

   ```bash
   python - <<'PY'
   from pathlib import Path

   import pandas as pd

   meta = pd.read_csv('meta.csv', sep=';')
   ids = meta['id'].unique()[:12]
   train, val, test = ids[:8], ids[8:10], ids[10:12]

   out_dir = Path('data/single_signer/index')
   out_dir.mkdir(parents=True, exist_ok=True)

   for split, values in {'train': train, 'val': val, 'test': test}.items():
       pd.Series(values, name='video_id').to_csv(out_dir / f'{split}.csv', index=False)
   PY
   ```

4. **Ejecuta el entrenamiento demo** con la CLI empaquetada:

   ```bash
   python -m slt \
     --face-dir data/single_signer/processed/face \
     --hand-left-dir data/single_signer/processed/hand_l \
     --hand-right-dir data/single_signer/processed/hand_r \
     --pose-dir data/single_signer/processed/pose \
     --metadata-csv meta.csv \
     --train-index data/single_signer/index/train.csv \
     --val-index data/single_signer/index/val.csv \
     --work-dir work_dirs/single_signer_demo \
     --epochs 2 --batch-size 2 --sequence-length 32
   ```

   El comando crea `last.pt` y `best.pt` en el directorio indicado, junto con un
   `config.json` que captura los parámetros efectivos.

Para configuraciones más extensas utiliza `tools/train_slt_multistream_v9.py`.
La guía en `docs/train_slt_multistream_v9.md` cubre los parámetros disponibles y
las recomendaciones de optimización.

## Evaluación

Evalúa el checkpoint más reciente y genera predicciones alineadas con los textos
referencia del CSV.

```bash
python tools/eval_slt_multistream_v9.py \
  --checkpoint work_dirs/single_signer_demo/best.pt \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --eval-index data/single_signer/index/test.csv \
  --output-csv work_dirs/single_signer_demo/predictions/preds.csv
```

Además del CSV de predicciones, la carpeta de salida contiene `report.json` y
`report.csv` con BLEU, CHRF y CER cuando se instala el extra `metrics`.

> **Nota:** el decoder textual debe provenir de un checkpoint T5/BART entrenado
> para la tarea. El tamaño oculto (`d_model`) del checkpoint tiene que coincidir
> con el `--d-model` del encoder multi-stream; de lo contrario la carga del
> modelo fallará.

## Exportación y despliegue

Exporta el encoder multi-stream a formatos ligeros para integrarlo en demos en
vivo o servicios de inferencia.

```bash
python tools/export_onnx_encoder_v9.py \
  --checkpoint work_dirs/single_signer_demo/best.pt \
  --onnx exports/single_signer_encoder.onnx \
  --torchscript exports/single_signer_encoder.ts \
  --image-size 224 --sequence-length 64 --d-model 512
```

Valida los artefactos con `tools/demo_realtime_multistream.py` o
`tools/test_realtime_pipeline.py`. Ambos aceptan modelos TorchScript u ONNX y un
tokenizer de HuggingFace para reconstruir los textos.

## Métricas de control

Las pruebas automatizadas actúan como chequeos de humo para verificar el
pipeline completo:

| Escenario | Archivo de prueba | Resultado |
|-----------|-------------------|-----------|
| Entrenamiento sintético | `tests/training/test_short_loop.py` | Pérdida cae tras 3 épocas. |
| Exportación de encoder | `tests/test_export.py` | ONNX y TorchScript generados en disco. |
| Eval. multistream | `tests/tools/test_eval_slt_multistream_v9.py` | Predicciones y BLEU. |

Ejecuta `pytest`, `ruff check .`, `black --check .` y `mypy` antes de enviar
cambios para garantizar consistencia con la CI.

## Demos en tiempo real y pruebas offline

Los scripts en `tools/` permiten ejecutar el encoder y decoder entrenados sobre
videos en vivo o grabados:

- `tools/demo_realtime_multistream.py`: captura desde cámara web, realiza el
  seguimiento con MediaPipe y muestra la traducción en un overlay de OpenCV.
- `tools/test_realtime_pipeline.py`: procesa un archivo de vídeo y puede crear
  un MP4 anotado para depurar el pipeline sin cámara.

Ambos scripts aceptan modelos TorchScript u ONNX exportados desde el pipeline y
tokenizers de HuggingFace para decodificar el texto.

## Preentrenamiento con DINO/iBOT

`tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py` permiten generar
pesos auto-supervisados compatibles con `load_dinov2_backbone`. La guía en
`docs/pretraining.md` detalla las configuraciones disponibles y la integración
con `MultiStreamEncoder`.
