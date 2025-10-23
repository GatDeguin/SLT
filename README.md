# SLT — Pipeline multi-stream para `single_signer`

Este repositorio centraliza las utilidades para preparar datos, entrenar,
evaluar, exportar y monitorear el modelo multi-stream validado para el flujo
`single_signer`. El pipeline opera sobre el dataset `single_signer`, cuyo CSV de
subtítulos principal es `meta.csv`, y sirve como plantilla para reproducir
experimentos o adaptar el flujo a nuevas variantes.

## Tabla de contenidos

1. [Resumen del pipeline](#resumen-del-pipeline)
2. [Instalación](#instalación)
3. [Flujo recomendado de extremo a extremo](#flujo-recomendado-de-extremo-a-extremo)
4. [Preparación de datos](#preparación-de-datos-datasingle_signer)
5. [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
6. [Exportación y demos](#exportación-y-demos-en-tiempo-real)
7. [Preentrenamiento de backbones](#preentrenamiento-de-backbones)
8. [Control de calidad y pruebas](#control-de-calidad-y-pruebas)
9. [Estructura de carpetas](#estructura-de-carpetas)
10. [Soporte y aportes](#soporte-y-aportes)

## Resumen del pipeline

- **Paquete `slt/`**: expone el dataset `LsaTMultiStream`, el encoder
  `MultiStreamEncoder`, un decoder seq2seq y utilidades para datos, métricas y
  entrenamiento.
- **Pipeline ROI + keypoints**: integra la extracción de recortes, el alineado
  de keypoints/glosas y el entrenamiento con pérdidas combinadas MSKA. Los
  detalles paso a paso están en `docs/train_slt_multistream_v9.md`.
- **Herramientas `tools/`**: scripts para extracción de ROIs, entrenamiento
  completo, evaluación, exportación, validación de contratos y demos en tiempo
  real (`docs/operational_checklist.md` lista los pasos sugeridos para releases).
- **Documentación `docs/`**: contratos de datos, guías operativas, manuales de
  preentrenamiento y resúmenes de papers que motivan el enfoque multi-stream.
- **Tests `tests/`**: suites de humo que cubren datos sintéticos,
  exportaciones y ejecuciones rápidas del pipeline.

Consulta `docs/data_contract.md`, `docs/train_slt_multistream_v9.md` y
`docs/pretraining.md` para ampliar cada etapa. `tools/README.md` describe cada
script CLI disponible.

### Pipeline unificado ROI + keypoints

1. **Preprocesar ROIs y keypoints** con `tools/extract_rois_v2.py` y el pipeline
   de MediaPipe documentado en `docs/data_contract.md`. Asegúrate de exportar
   `metadata.jsonl` y los keypoints por video (`.npz`/`.npy`).
2. **Validar alineación y glosas** ejecutando
   `python tools/ci_validate_data_contract.py`, que comprueba máscaras,
   keypoints y etiquetas CTC/gloss en un dataset sintético.
3. **Entrenar con pérdidas combinadas** mediante
   `tools/train_slt_multistream_v9.py --use-mska`, definiendo los pesos de
   traducción, CTC y distilación según el escenario (ver ejemplos en
   `tools/README.md`). El wrapper `tools/train_slt_lsa_mska_v13.py` mantiene
   compatibilidad con scripts heredados.
4. **Evaluar y exportar** con `tools/eval_slt_multistream_v9.py` y
   `tools/export_onnx_encoder_v9.py`, reutilizando las mismas configuraciones de
   MSKA para reproducir métricas y artefactos.

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
aplica los requisitos de desarrollo. `requirements-dev.txt` instala el paquete
en modo editable junto con herramientas de linting y pruebas.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows usa .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Extras opcionales

El paquete define grupos de extras que habilitan funcionalidades específicas:

| Extra | Propósito |
|-------|-----------|
| `media` | Seguimiento facial/manos con MediaPipe para `tools/extract_rois_v2.py`. |
| `metrics` | Cálculo de BLEU, ChrF, CER y WER durante la evaluación. |
| `export` | Exportación ONNX/TorchScript y validación en tiempo real. |

Instálalos en bloque con `pip install .[media,metrics,export]` o agrega cada uno
según tus necesidades. Consulta la sección de [control de calidad](#control-de-calidad-y-pruebas)
para conocer las verificaciones recomendadas tras la instalación.

## Flujo recomendado de extremo a extremo

1. **Preparar datos** siguiendo el [contrato documentado](docs/data_contract.md).
   Ejecuta `tools/extract_rois_v2.py` sobre los videos y construye los splits en
   `data/single_signer/index/`.
2. **Verificar la instalación** corriendo `pytest` y los linters (`ruff`,
   `black`, `mypy`) para confirmar que el entorno está consistente.
3. **Ejecutar un entrenamiento rápido** con `python -m slt` para validar que los
   datos y el tokenizador son correctos. Define el `work_dir` donde se guardarán
   los checkpoints temporales.
4. **Lanzar experimentos completos** con `tools/train_slt_multistream_v9.py`,
   habilitando `--use-mska` cuando el dataset incluya keypoints y glosas. El
   wrapper `tools/train_slt_lsa_mska_v13.py` reenvía los argumentos al flujo
   unificado.
5. **Evaluar resultados** usando `tools/eval_slt_multistream_v9.py` y analiza los
   reportes con `docs/metrics_dashboard_integration.py` o tus dashboards.
6. **Exportar y validar** con `tools/export_onnx_encoder_v9.py` y las demos en
   tiempo real (`tools/demo_realtime_multistream.py`,
   `tools/test_realtime_pipeline.py`).
7. **Documentar y publicar** apoyándote en `docs/operational_checklist.md` antes
   de liberar artefactos o abrir un PR.

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
         keypoints/
       annotations/
         gloss.csv
       index/
         train.csv
         val.csv
         test.csv
   meta.csv              # CSV con columnas video_id;texto
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
   fuese necesario. Las poses se guardan normalizadas en `[0, 1]` dentro del
   *signing space* (ancho 6, alto 7 unidades de cabeza) y los `.npz` incluyen la
   clave `pose_norm="signing_space_v1"`. Cuando MediaPipe no reporta landmarks
   se replica la pose previa o se rellena con `-1` y visibilidad `0` como
   sentinel.
4. Genera o copia los keypoints multistream en
   `data/single_signer/processed/keypoints/`. Cada archivo debe nombrarse como
   `<video_id>.npy` o `<video_id>.npz` y contener un arreglo `keypoints` en
   formato `(T, landmarks, 3)` con `(x, y, conf)` siguiendo el layout de
   MediaPipe. Si cuentas con anotaciones de glosa, agrégalas al CSV opcional
   `data/single_signer/annotations/gloss.csv` con columnas
   `video_id;gloss;ctc_labels` (índices separados por espacios).
5. Construye los splits desde `meta.csv` según tus criterios. Los CSV deben
   contener un `video_id` por línea sin encabezado. Un ejemplo mínimo:
   ```bash
   python - <<'PY'
   from pathlib import Path
   import pandas as pd

   meta = pd.read_csv('meta.csv', sep=';')
   ids = meta['video_id'].unique()
   out_dir = Path('data/single_signer/index')
   out_dir.mkdir(parents=True, exist_ok=True)
   pd.Series(ids[:80]).to_csv(out_dir / 'train.csv', index=False)
   pd.Series(ids[80:90]).to_csv(out_dir / 'val.csv', index=False)
   pd.Series(ids[90:]).to_csv(out_dir / 'test.csv', index=False)
   PY
   ```
6. Valida la estructura con `python tools/ci_validate_data_contract.py` o
   ejecuta `pytest tests/data/test_dataset_quality.py` para comprobar los
   avisos producidos ante inconsistencias de FPS, frames faltantes, keypoints
   ausentes o splits incompletos.

El contrato de datos completo se detalla en `docs/data_contract.md`, donde se
documentan los campos opcionales de `meta.csv`, las métricas registradas en
`metadata.jsonl` y las convenciones de nomenclatura de archivos.

## Entrenamiento y evaluación

### Entrenamiento rápido (`python -m slt`)

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
  --keypoints-dir data/single_signer/processed/keypoints \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --gloss-csv data/single_signer/annotations/gloss.csv \
  --work-dir work_dirs/single_signer_demo \
  --epochs 2 --batch-size 2 --sequence-length 32 \
  --tokenizer hf-internal-testing/tiny-random-T5
```

El comando guarda `last.pt`, `best.pt` y `config.json` dentro de `work_dir`.

### Entrenamiento completo multi-stream

Para sesiones prolongadas utiliza `tools/train_slt_multistream_v9.py`, que
expone reanudación, *mixup* por stream, configuración vía YAML/JSON y
sobrescritura puntual con `--set`. La guía detallada está en
`docs/train_slt_multistream_v9.md`.

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set data.face_dir=data/single_signer/processed/face \
  --set data.work_dir=work_dirs/single_signer_experiment \
  --set training.epochs=40 \
  --set optim.lr=5e-4
```

Los templates aceptan ahora `data.keypoints_dir` y `data.gloss_csv` para
propagar keypoints MediaPipe y secuencias de glosa/CTC al pipeline sin editar el
código base.

`config.json` dentro de `work_dir` refleja la configuración efectiva combinando
defaults, archivo y banderas.

### Entrenamiento basado en keypoints

Activa la rama MSKA con `tools/train_slt_multistream_v9.py --use-mska`,
proporcionando `data.keypoints_dir` y `data.gloss_csv` en la configuración.
Esto habilita la combinación de pérdidas (traducción, CTC y distilación) y la
carga de checkpoints individuales para el encoder y las cabezas auxiliares.
El script `tools/train_slt_lsa_mska_v13.py` queda como *wrapper* retrocompatible
que reenvía los parámetros al flujo unificado.

```bash
python tools/train_slt_multistream_v9.py \
  --use-mska \
  --keypoints-dir data/lsa_keypoints \
  --gloss-csv data/lsa_gloss.csv \
  --work-dir work_dirs/lsa_mska \
  --mska-ctc-weight 0.5 --mska-distillation-weight 0.2
```

Ajusta la atención por articulación y los bloques temporales con
`--mska-stream-heads`, `--mska-temporal-blocks`, `--mska-temporal-kernel` y
`--mska-temporal-dilation`, tal
como se describe en la Sección 3.2.2 del paper y en
`docs/train_slt_multistream_v9.md`.

El refinamiento global descrito en la Sección 3.2.4 se activa con
`--mska-use-sgr`. Controla su aportación con `--mska-sgr-mix`, selecciona la
activación mediante `--mska-sgr-activation` y decide si la matriz se comparte
entre streams (`--mska-sgr-shared`) o se aprende por flujo (`--mska-sgr-per-stream`).

Revisa `tools/train_slt_multistream_v9.py --help` y
`docs/train_slt_multistream_v9.md` para detalles de cada argumento.

### Evaluación y reportes

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

El script valida el tokenizador con `slt.utils.validate_tokenizer` antes de
procesar los videos, evitando ejecuciones largas con configuraciones inválidas.
Los reportes incluyen métricas BLEU, ChrF, CER y WER; estas últimas se calculan
mediante `slt.utils.character_error_rate` y `slt.utils.word_error_rate` para
facilitar comparaciones con referencias externas.

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
`tools/test_realtime_pipeline.py` (video en disco). Ambos requieren un modelo
exportado (`--model`) o declarar `--model-format stub`, además de un tokenizador
de HuggingFace (`--tokenizer`).

## Preentrenamiento de backbones

`tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py` permiten obtener
pesos auto-supervisados compatibles con `load_dinov2_backbone`. Soportan
entrenamientos declarativos vía `--config`, exportación de backbones (`--export-backbone`)
y registro de métricas en `metrics.jsonl`. Sigue la guía en `docs/pretraining.md`
para elegir hiperparámetros, controlar augmentations y conectar los pesos
resultantes con `tools/train_slt_multistream_v9.py`. La misma guía detalla cómo
activar el regularizador KoLeo mediante `--koleo-weight` y ajustar la estabilidad
numérica con `--koleo-epsilon`.

## Control de calidad y pruebas

Los tests automatizados validan piezas clave del pipeline:

| Escenario | Prueba | Resultado |
|-----------|--------|-----------|
| Contrato datos | `python tools/ci_validate_data_contract.py` | Replica estructura base. |
| E2E sintético | `tests/test_pipeline_end_to_end.py` | Pérdida cae y exporta. |
| CLI demo | `tests/test_cli_main.py` | Entrenamiento corto sin errores. |
| Exportación | `tests/test_export.py` | ONNX y TorchScript válidos. |
| Calidad datos | `tests/data/test_dataset_quality.py` | Detecta frames faltantes y FPS. |
| Métricas regresión | `python tools/ci_validate_metrics.py` | Pérdidas dentro de tolerancias. |

Ejecuta `pytest`, `ruff check .`, `black --check .` y `mypy` antes de subir
cambios. `docs/operational_checklist.md` resume la secuencia recomendada de
verificaciones previas a una release.

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
