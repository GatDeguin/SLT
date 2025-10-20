# SLT — Pipeline multi-stream para `single_signer`

Este repositorio agrupa las utilidades necesarias para preparar datos, entrenar,
evaluar y exportar el modelo multi-stream validado para el flujo `single_signer`
del paquete `slt`. Los scripts están pensados como un flujo de referencia sobre
el dataset `single_signer`, cuyo CSV de subtítulos principal es `meta.csv`.

## Contenido del repositorio

- Paquete `slt/` con el dataset `LsaTMultiStream`, el encoder
  `MultiStreamEncoder`, envoltorios de entrenamiento y funciones auxiliares.
- Herramientas en `tools/` para extracción de ROIs, entrenamiento completo,
  evaluación, exportación y demos en tiempo real.
- Documentación detallada en `docs/` con contratos de datos, guías operativas y
  manuales de preentrenamiento.
- Tests de humo en `tests/` que cubren el pipeline extremo a extremo utilizando
  datasets sintéticos.

Consulta `docs/data_contract.md`, `docs/train_slt_multistream_v9.md` y
`docs/pretraining.md` para ampliar cada etapa.

### Pesos pre-entrenados

El repositorio no incluye el checkpoint validado para `single_signer` debido a
restricciones de tamaño. Descarga `single_signer_multistream.pt` desde la
ubicación compartida por el equipo y colócalo en
`data/single_signer/single_signer_multistream.pt` o expón su ruta mediante la
variable de entorno `SLT_SINGLE_SIGNER_CHECKPOINT`. Todas las CLI buscarán el
archivo siguiendo ese orden; puedes desactivarlo con `--pretrained none`. El
encoder puede instanciarse de forma directa con:

```python
from slt.models import MultiStreamEncoder

encoder = MultiStreamEncoder.from_pretrained(
    "single_signer",
    checkpoint_path="data/single_signer/single_signer_multistream.pt",
)
```

## Instalación

Configura un entorno virtual, instala PyTorch compatible con tu hardware y
aplica los requisitos de desarrollo.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows usa .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` instala el paquete en modo editable junto con las
siguientes dependencias opcionales:

| Extra | Propósito |
|-------|-----------|
| `media` | Seguimiento facial/manos con MediaPipe para `tools/extract_rois_v2.py`. |
| `metrics` | Cálculo de BLEU, ChrF, CER y WER durante la evaluación. |
| `export` | Exportación ONNX/TorchScript y validación en tiempo real. |

También puedes instalarlos en pasos separados con `pip install .[media,metrics,export]`.

## Preparación de datos (`data/single_signer`)

1. Crea la estructura base dentro de `data/single_signer/`:
   ```text
   data/
     single_signer/
       videos/           # Clips fuente en MP4/MKV
       processed/
         face/
         hand_l/
         hand_r/
         pose/
       index/
         train.csv
         val.csv
         test.csv
   meta.csv              # CSV con columnas id;texto
   ```
2. Copia los videos originales en `data/single_signer/videos/`.
3. Ejecuta la extracción de regiones de interés para rostro, manos y pose:
   ```bash
   python tools/extract_rois_v2.py \
     --videos data/single_signer/videos \
     --output data/single_signer/processed \
     --metadata meta.csv
   ```
   El script genera `face/`, `hand_l/`, `hand_r/` y `pose/` junto a un
   `metadata.jsonl` con métricas por video. Reanuda ejecuciones con `--resume` si
   fuese necesario.
4. Construye los splits desde `meta.csv` según tus criterios. Un ejemplo mínimo:
   ```bash
   python - <<'PY'
   from pathlib import Path
   import pandas as pd

   meta = pd.read_csv('meta.csv', sep=';')
   ids = meta['id'].unique()
   out_dir = Path('data/single_signer/index')
   out_dir.mkdir(parents=True, exist_ok=True)
   pd.Series(ids[:80]).to_csv(out_dir / 'train.csv', index=False)
   pd.Series(ids[80:90]).to_csv(out_dir / 'val.csv', index=False)
   pd.Series(ids[90:]).to_csv(out_dir / 'test.csv', index=False)
   PY
   ```

El contrato de datos completo se detalla en `docs/data_contract.md`.

## Entrenamiento rápido (`python -m slt`)

La CLI empaquetada ejecuta un entrenamiento corto para verificar el pipeline.
Por defecto intenta inicializar con los pesos `single_signer` descargados, por
lo que puedes comenzar a afinar el modelo directamente. Si el checkpoint vive en
una ruta distinta utiliza `--pretrained-checkpoint /ruta/al/archivo.pt`. Para
reiniciar desde parámetros aleatorios añade `--pretrained none`.

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
  --epochs 2 --batch-size 2 --sequence-length 32 \
  --tokenizer hf-internal-testing/tiny-random-T5
```

El comando guarda `last.pt`, `best.pt` y `config.json` dentro de `work_dir`.

Para sesiones largas utiliza `tools/train_slt_multistream_v9.py`, que expone
reanudación, *mixup* por stream y compatibilidad con plantillas YAML/JSON.
Consulta `docs/train_slt_multistream_v9.md` para conocer todos los parámetros.

## Evaluación

Evalúa uno o varios checkpoints y genera predicciones, métricas y reportes. Para
usar el preset validado sin un checkpoint propio pasa `--checkpoint single_signer`
y, si el archivo no vive en `data/single_signer/`, añade
`--pretrained-checkpoint /ruta/al/archivo.pt`:

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

Los archivos `report.json` y `report.csv` resultantes pueden integrarse en
herramientas analíticas mediante `docs/metrics_dashboard_integration.py`.

## Exportación y demos en tiempo real

Convierte el encoder a ONNX o TorchScript para ejecutarlo fuera de PyTorch. El
ejemplo siguiente exporta el preset `single_signer`; añade
`--pretrained-checkpoint /ruta/al/archivo.pt` si el checkpoint descargado no se
encuentra en `data/single_signer/`:

```bash
python tools/export_onnx_encoder_v9.py \
  --checkpoint single_signer \
  --onnx exports/single_signer_encoder.onnx \
  --torchscript exports/single_signer_encoder.ts \
  --sequence-length 64
```

Valida los artefactos con `tools/demo_realtime_multistream.py` (webcam) o
`tools/test_realtime_pipeline.py` (video en disco). Ambos aceptan modelos
TorchScript/ONNX y tokenizadores de HuggingFace. Si no se especifica un modelo,
utilizarán el preset validado (asegúrate de tener el checkpoint descargado).

## Preentrenamiento de backbones

`tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py` permiten obtener
pesos auto-supervisados compatibles con `load_dinov2_backbone`. Sigue la guía en
`docs/pretraining.md` para exportar los backbones y conectarlos con el flujo de
entrenamiento principal.

## Control de calidad y pruebas

Los tests automatizados validan piezas clave del pipeline:

| Escenario | Prueba | Resultado esperado |
|-----------|--------|--------------------|
| Datos sintéticos end-to-end | `tests/test_pipeline_end_to_end.py` | Pérdida cae y exporta encoder. |
| CLI de demo | `tests/test_cli_main.py` | Ejecuta entrenamiento corto sin errores. |
| Exportación | `tests/test_export.py` | Genera y valida ONNX/TorchScript. |
| Calidad de datos | `tests/data/test_dataset_quality.py` | Detecta frames faltantes y FPS. |

Ejecuta `pytest`, `ruff check .`, `black --check .` y `mypy` antes de subir
cambios.

## Estructura de carpetas

```text
slt/                 # Paquete instalable con encoder, dataset y utilidades
  data/
  models/
  training/
  runtime/
  utils/
tools/               # Scripts CLI para extracción, entrenamiento, evaluación, demos
docs/                # Guías de datos, operación, preentrenamiento y métricas
tests/               # Suite de smoke tests y fixtures sintéticas
meta.csv             # CSV de subtítulos de `single_signer`
```

## Soporte y aportes

Abre *issues* o PRs describiendo claramente el componente afectado (datos,
entrenamiento, exportación, demos). Acompaña los cambios con actualizaciones en
la documentación cuando modifiques rutas, argumentos o artefactos generados.
