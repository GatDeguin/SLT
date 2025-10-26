# Visión general de la arquitectura

Esta guía describe cómo se conectan los paquetes `slt/`, las CLI en `tools/` y los
artefactos auxiliares para reproducir el pipeline multi-stream documentado en el
README. Emplea rutas relativas a la raíz del repositorio y se complementa con
`docs/data_contract.md`, `docs/pretraining.md` y `docs/operational_checklist.md`.

## Flujo de datos extremo a extremo

1. **Fuentes de video y metadata**: los clips viven en `data/single_signer/videos/`
   y sus transcripciones en `meta.csv`.
2. **Extracción**: `tools/extract_rois_v2.py` genera recortes por stream, keypoints
   y `metadata.jsonl` respetando el contrato de datos.
3. **Dataset instalable**: `slt/data/lsa_t_multistream.py` crea tensores, máscaras y
   metadatos listos para entrenamiento.
4. **Encoder multi-stream**: `slt/models/multistream.py` fusiona rostro, manos y pose.
5. **Decodificador y pérdidas**: `slt/training/models.py` arma el modelo final con
   MSKA opcional (`slt/models/mska.py`).
6. **Entrenamiento**: `slt/training/loops.py` y `slt/training/evaluation.py`
   implementan los bucles de optimización y validación.
7. **Exportación y demos**: `tools/export_onnx_encoder_v9.py` y los scripts de
   tiempo real usan las utilidades de `slt/runtime/` para preparar inferencia.

## Paquete `slt/`

### Datos (`slt/data`)

- `LsaTMultiStream` prepara tensores para rostro, manos, pose y keypoints,
  aplicando augmentations y controles de calidad configurables.
- `collate_fn` agrupa lotes e incorpora máscaras para cada stream, listas para
  alimentar el encoder multi-stream y el decoder de secuencia a secuencia.
- El módulo expone funciones auxiliares para normalizar rangos de augmentations y
  reorganizar el layout de keypoints siguiendo MediaPipe.

### Modelos (`slt/models`)

- `MultiStreamEncoder` proyecta cada stream, concatena las representaciones y las
  procesa con `TemporalEncoder` antes de delegar en MSKA o el decoder principal.
- `mska.py` define el encoder de atención multi-stream y su salida (`MSKAOutput`).
- `backbones.py` carga pesos preentrenados (DINO/iBOT o checkpoints propios) y
  permite congelar ramas individuales.
- `temporal.py` expone bloques transformadores temporales con máscaras de padding
  consistentes con `LsaTMultiStream`.

### Entrenamiento (`slt/training`)

- `configuration.py` declara las `dataclass` `DataConfig`, `ModelConfig`,
  `OptimConfig` y `TrainingConfig` consumidas por las CLI (`resolve_configs`).
- `data.py` crea `DataLoader`, aplica mezcla de streams y prepara tokenizadores.
- `models.py` ensambla el encoder, el decoder seq2seq y las cabezas MSKA.
- `loops.py` implementa `train_epoch`/`eval_epoch` con gradiente acumulado,
  recortes automáticos y soporte para `torch.compile`.
- `evaluation.py` calcula métricas BLEU/ChrF/CER/WER y reportes estructurados.
- `optim.py` inicializa optimizadores, planificadores y clipping de gradiente.

### Utilidades y runtime

- `slt/__main__.py` ejecuta `python -m slt` como demo rápida reutilizando los
  mismos componentes que `tools/train_slt_multistream_v9.py`.
- `slt/runtime/realtime.py` contiene `TemporalBuffer` y helpers para demos con
  latencia controlada.
- `slt/utils/cli.py` ofrece parsers para rangos flotantes que utilizan todas las
  CLI (augmentations de keypoints, mezclas de streams, etc.).
- `slt/utils/general.py` centraliza semillas, sincronización de dispositivos y
  utilidades de logging.
- `slt/utils/text.py` resuelve tokenizadores Hugging Face, caches locales y
  prompts aprendibles.

## CLI y herramientas en `tools/`

- **Preparación**: `extract_rois_v2.py`, `prepare_lsat_crops.py` y
  `ci_validate_data_contract.py` aseguran que el dataset cumpla el contrato.
- **Entrenamiento**: `train_slt_multistream_v9.py` expone todas las banderas; el
  wrapper `train_slt_lsa_mska_v13.py` preserva compatibilidad heredada.
- **Evaluación y monitoreo**: `eval_slt_multistream_v9.py`,
  `ci_validate_metrics.py` y `docs/metrics_dashboard_integration.py` generan
  reportes comparables en CI.
- **Exportación y demos**: `export_onnx_encoder_v9.py`,
  `demo_realtime_multistream.py` y `test_realtime_pipeline.py` verifican
  inferencia en tiempo real con modelos exportados o stub.

Cada script acepta `--help` y admite configuraciones declarativas (`--config`) y
sobrescrituras con `--set clave=valor` para reproducibilidad.

## Configuraciones y presets

- Los presets de decoder (`configs/presets/*.yaml`) sincronizan tokenizadores,
  arquitecturas y pesos iniciales. Usa `--decoder-preset <nombre>` o
  `--decoder-model` para alternar entre MBART y T5 según el escenario.
- `configs/*.yml` muestra ejemplos de `DataConfig` y `ModelConfig` para datasets
  Phoenix, LSA-T y SignMusketeers. Ajusta rutas y pesos antes de lanzar cada
  experimento.
- `docs/mska-paper-config.md` resume la configuración exacta del paper y cómo
  reproducirlo con los scripts actuales.

## Calidad, pruebas y monitoreo

- Ejecuta `pytest`, `ruff check .`, `black --check .` y `mypy` como parte del
  checklist descrito en `docs/operational_checklist.md`.
- `tests/` contiene suites de humo para datos, exportación, demo CLI y fin a fin
  con datasets sintéticos (`tests/data/test_dataset_quality.py`,
  `tests/test_pipeline_end_to_end.py`, etc.).
- `tools/ci_validate_metrics.py` compara pérdidas con valores de referencia y
  alerta sobre desviaciones significativas.
- `docs/metrics_dashboard_integration.py` ofrece un script de ejemplo para
  consolidar métricas en dashboards externos.

## Trabajo offline y caches

- `docs/offline_cache.md` explica cómo generar caches de tokenizadores y decoders
  cuando no hay acceso a internet. Las CLI aceptan `--tokenizer-search-path` y
  `--decoder-search-path` para reutilizar los artefactos.
- Define `SLT_SINGLE_SIGNER_CHECKPOINT` y `SLT_PHOENIX_CHECKPOINT` para evitar
  repetir banderas en entornos con rutas personalizadas.

## Próximos pasos

- Mantén sincronizados README y docs tras modificar rutas, presets o argumentos.
- Anota en commits y PRs los componentes afectados (datos, entrenamiento,
  exportación, demos) para facilitar revisiones cruzadas.
