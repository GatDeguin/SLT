# Herramientas

Este directorio reúne las CLI que complementan el paquete `slt`. Todas requieren
Python 3.9+ y las dependencias del proyecto instaladas en modo editable.

## Dependencias opcionales

```bash
pip install .[media,metrics,export]
```

- `media`: detectores de MediaPipe para `extract_rois_v2.py` y demos en vivo.
- `metrics`: métricas BLEU/ChrF/CER/WER utilizadas por `eval_slt_multistream_v9.py`.
- `export`: soporte de ONNX y utilidades de validación en tiempo real.

`opencv-python` está incluido en los requisitos base. Instala el extra `media`
cuando necesites procesar videos o ejecutar demos.

## Preparación y validación de datos

- `extract_rois_v2.py`: genera los recortes de rostro/manos/pose en
  `data/single_signer/processed/` siguiendo el [contrato de datos](../docs/data_contract.md).
  Soporta reanudación con `--resume` y produce `metadata.jsonl` con métricas por video.
- `ci_validate_data_contract.py`: construye un dataset sintético que replica la
  estructura esperada y ejecuta verificaciones básicas.
- `ci_validate_metrics.py`: compara pérdidas y métricas contra valores de
  referencia para detectar regresiones en CI.

## Entrenamiento

- `python -m slt`: demo interactiva que ejecuta un entrenamiento corto. Útil
  para validar dependencias y rutas antes de lanzar experimentos prolongados.
- `train_slt_multistream_v9.py`: entrenamiento completo multi-stream. Admite
  configuración declarativa (`--config`), sobrescrituras (`--set`) y mezcla de
  streams (`--mix-stream`). Consulta [la guía dedicada](../docs/train_slt_multistream_v9.md).
- `train_slt_lsa_mska_v13.py`: *wrapper* de compatibilidad que reenvía los
  argumentos a `train_slt_multistream_v9.py --use-mska`. Utiliza el flujo unificado
  para entrenar con keypoints y pérdidas auxiliares configurables.

### Parámetros clave para MSKA

- `--keypoints-dir`: habilita la lectura de keypoints MediaPipe (`.npz`/`.npy`).
- `--gloss-csv`: propaga las secuencias de glosa y etiquetas CTC al entrenamiento.
- `--use-mska`: activa la rama de atención multi-stream sobre keypoints.
- `--mska-translation-weight`, `--mska-ctc-weight`, `--mska-distillation-weight`:
  controlan la combinación de pérdidas.
- `--mska-distillation-temperature`: temperatura aplicada al término de
  distilación.
- `--mska-ctc-vocab`, `--mska-input-dim`, `--mska-detach-teacher`: parámetros
  estructurales del encoder MSKA.

### Ejemplos de configuración con MSKA

```yaml
data:
  face_dir: data/single_signer/processed/face
  hand_left_dir: data/single_signer/processed/hand_l
  hand_right_dir: data/single_signer/processed/hand_r
  pose_dir: data/single_signer/processed/pose
  keypoints_dir: data/single_signer/processed/keypoints
  metadata_csv: meta.csv
  train_index: data/single_signer/index/train.csv
  val_index: data/single_signer/index/val.csv
  gloss_csv: data/single_signer/annotations/gloss.csv
  work_dir: work_dirs/slt_mska
  tokenizer: hf-internal-testing/tiny-random-T5
  max_target_length: 64
model:
  use_mska: true
  mska_ctc_vocab: 64
  mska_translation_weight: 0.7
  mska_ctc_weight: 0.2
  mska_distillation_weight: 0.1
  mska_distillation_temperature: 2.0
training:
  epochs: 30
optim:
  lr: 0.0005
```

```json
{
  "data": {
    "face_dir": "data/single_signer/processed/face",
    "hand_left_dir": "data/single_signer/processed/hand_l",
    "hand_right_dir": "data/single_signer/processed/hand_r",
    "pose_dir": "data/single_signer/processed/pose",
    "keypoints_dir": "data/single_signer/processed/keypoints",
    "metadata_csv": "meta.csv",
    "train_index": "data/single_signer/index/train.csv",
    "val_index": "data/single_signer/index/val.csv",
    "gloss_csv": "data/single_signer/annotations/gloss.csv",
    "work_dir": "work_dirs/slt_mska",
    "tokenizer": "hf-internal-testing/tiny-random-T5",
    "max_target_length": 64
  },
  "model": {
    "use_mska": true,
    "mska_ctc_vocab": 64,
    "mska_translation_weight": 0.7,
    "mska_ctc_weight": 0.2,
    "mska_distillation_weight": 0.1,
    "mska_distillation_temperature": 2.0
  },
  "training": {
    "epochs": 30
  },
  "optim": {
    "lr": 0.0005
  }
}
```

## Evaluación y exportación

- `eval_slt_multistream_v9.py`: valida tokenizadores, calcula métricas BLEU,
  ChrF, CER y WER, y genera `report.json`/`report.csv` para dashboards.
- `export_onnx_encoder_v9.py`: produce artefactos ONNX y TorchScript y puede
  integrarse con los scripts de demos.
- `demo_realtime_multistream.py`: demostración con webcam que consume modelos
  exportados (`--model`) u opciones `--model-format stub` para pruebas sin
  pesos. Requiere GPU para el mejor rendimiento.
- `test_realtime_pipeline.py`: reproduce las demos con videos grabados y puede
  guardar anotaciones de salida (`--output`).

## Utilidades adicionales

- `_pretrain_dino.py`, `pretrain_dino_face.py`, `pretrain_dino_hands.py`:
  scripts para entrenar backbones auto-supervisados DINO/iBOT. Revisa
  [docs/pretraining.md](../docs/pretraining.md) para flujos avanzados.
- `pretrain_utils.py`: helpers compartidos para manejo de configuraciones,
  logging y exportación de pesos.

Cada CLI acepta `--help` para consultar todos los argumentos disponibles.
