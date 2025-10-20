# SLT — Demo de entrenamiento

Este repositorio incluye una implementación modular del stub multi-stream
utilizado durante la experimentación con el corpus LSA-T. El comando
`python -m slt` reproduce el ejemplo del archivo "Proyecto": crea el dataset,
los *DataLoaders* y ejecuta un entrenamiento corto con los modelos de prueba
incluidos en el paquete.

## Instalación

El proyecto se distribuye como un paquete editable. Para un entorno de
desarrollo completo instala las dependencias de la siguiente manera:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows utiliza `.venv\\Scripts\\activate`
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Esto instalará el paquete `slt`, las dependencias de ejecución (PyTorch,
Transformers, Pandas, etc.) y las herramientas de desarrollo utilizadas en CI
(`pytest`, `ruff`, `black`, `mypy`, `onnx`). Para trabajar con la extracción de
regiones y la demo en tiempo real puedes instalar extras opcionales:

```bash
pip install .[media]     # MediaPipe para extracción de ROI
pip install .[export]    # Dependencias para exportar a ONNX/TorchScript
```

## Preparación de datos

La demo y los scripts de entrenamiento esperan la estructura generada por
`tools/extract_rois_v2.py`. El flujo típico es:

1. Ejecutar `tools/extract_rois_v2.py` sobre los videos brutos para generar las
   carpetas `face/`, `hand_l/`, `hand_r/` y `pose/`.
2. Generar el CSV principal `subs.csv` con columnas `video_id;texto`.
3. Crear dos CSV adicionales (`train.csv` y `val.csv`) con la lista de
   identificadores utilizados en cada split.

```bash
python tools/extract_rois_v2.py \
  --videos data/raw_videos \
  --output data/rois \
  --metadata meta.csv
```

El dataset multi-stream (`slt.data.LsaTMultiStream`) validará la presencia de
las columnas requeridas y normalizará automáticamente los streams de imagen,
pose y máscaras de confianza.

## Entrenamiento

```bash
python -m slt \
  --face-dir data/rois/face \
  --hand-left-dir data/rois/hand_l \
  --hand-right-dir data/rois/hand_r \
  --pose-dir data/rois/pose \
  --metadata-csv data/lsa_t/subs.csv \
  --train-index data/lsa_t/index/train.csv \
  --val-index data/lsa_t/index/val.csv \
  --work-dir work_dirs/demo \
  --batch-size 2 --epochs 2
```

El script guardará `last.pt` y `best.pt` en `--work-dir` y mostrará en consola
la pérdida de entrenamiento/validación por época. Ajusta los parámetros según
la disponibilidad de hardware (por ejemplo `--device cpu` para forzar la
ejecución en CPU). Si deseas un control más detallado del pipeline utiliza el
script `tools/train_slt_multistream_v9.py`, que expone opciones adicionales
para optimización, *logging* y reanudación de checkpoints:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/demo.json \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --face-dir data/rois/face \
  --train-index data/lsa_t/index/train.csv \
  --val-index data/lsa_t/index/val.csv \
  --epochs 40 --batch-size 4
```

Consulta `docs/train_slt_multistream_v9.md` para una referencia completa de
argumentos y buenas prácticas.

## Evaluación

El script `tools/eval_slt_multistream_v9.py` calcula métricas de traducción
utilizando los checkpoints generados durante el entrenamiento. Un ejemplo de
uso es:

```bash
python tools/eval_slt_multistream_v9.py \
  --checkpoint work_dirs/multistream_v9/best.pt \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --face-dir data/rois/face \
  --metadata-csv data/lsa_t/subs.csv \
  --index data/lsa_t/index/test.csv
```

El script reporta pérdida promedio y, si se proporcionan referencias, métricas
como CER y BLEU mediante `sacrebleu`.

## Exportación y despliegue

Para desplegar el encoder multi-stream en aplicaciones móviles o backends
livianos, exporta a ONNX y TorchScript con `tools/export_onnx_encoder_v9.py`:

```bash
python tools/export_onnx_encoder_v9.py \
  --checkpoint work_dirs/multistream_v9/best.pt \
  --onnx exports/encoder.onnx \
  --torchscript exports/encoder.ts \
  --image-size 224 --sequence-length 64 --d-model 512
```

Luego utiliza `tools/demo_realtime_multistream.py` o
`tools/test_realtime_pipeline.py` para validar el modelo exportado en una demo
de cámara web o sobre videos pregrabados. Ambos scripts aceptan modelos
TorchScript/ONNX y un tokenizer de HuggingFace para decodificar el texto.

## Métricas esperadas

La siguiente tabla resume valores de referencia obtenidos con los stubs
incluidos en el repositorio. Funcionan como chequeos de humo para validar que
el entorno está correctamente configurado.

| Escenario | Métrica | Valor esperado |
|-----------|---------|----------------|
| Entrenamiento lineal sintético (`tests/training/test_short_loop.py`) | Pérdida inicial (`eval_epoch`) | ≈ 16.10 |
| Entrenamiento lineal sintético (`tests/training/test_short_loop.py`) | Pérdida final tras 3 épocas | ≈ 0.57 |
| Exportación encoder (`tests/test_export.py`) | Archivos generados | `encoder_*.onnx`, `encoder_*.ts` |

## Sustituir los stubs por modelos reales

Los componentes incluidos en el paquete están pensados para ser reemplazados
por modelos de producción:

1. **Backbones / proyectores**: `slt/models/backbones.py` expone
   `ViTSmallPatch16` como un stub ligero y `slt/models/modules.py` define
   `StreamProjector` y `FuseConcatLinear`. Puedes extenderlos o sobrescribirlos
   para cargar pesos de DINOv2 y proyectores oficiales (método
   `replace_with_dinov2`).
2. **Encoder multi-stream**: `slt/models/multistream.py` centraliza la lógica
   de fusión y temporales. Sustituye sus dependencias por las variantes reales
   y ajusta la máscara de manos si cuentas con detección de frames perdidos.
3. **Decoder textual**: `slt/models/temporal.py` implementa
   `TextSeq2SeqDecoder`, una envoltura sobre modelos seq2seq de HuggingFace.
   Ajusta la configuración (tokenizer, arquitectura, longitud máxima) para
   cargar tu modelo de producción o inicializar pesos pre-entrenados.

Una vez actualizados estos módulos, la demo servirá como punto de partida para
un pipeline de entrenamiento completo con pesos reales.

## Demos en tiempo real y pruebas offline

Los scripts en `tools/` permiten ejecutar el encoder/decoder entrenado sobre secuencias capturadas en vivo o videos pregrabados:
- `tools/demo_realtime_multistream.py`: captura desde cámara web, realiza el tracking de rostro/manos con MediaPipe y muestra la traducción en un overlay de OpenCV.
- `tools/test_realtime_pipeline.py`: procesa un archivo de video y opcionalmente genera un MP4 anotado para depurar el pipeline sin cámara.

Ambos scripts aceptan modelos TorchScript u ONNX exportados desde el pipeline de entrenamiento y, si se proporciona un tokenizador de HuggingFace (`--tokenizer`), decodifican el texto completo en consola y en el overlay.

## Preentrenamiento con DINO/iBOT

Los scripts `tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py`
permiten generar pesos auto-supervisados compatibles con
`load_dinov2_backbone`. Consulta la guía en `docs/pretraining.md` para conocer
las opciones disponibles (DINO/iBOT, *multi-crop*, *EMA*, exportación de
backbones) y cómo integrarlos con `MultiStreamEncoder`.
