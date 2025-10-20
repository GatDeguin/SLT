# Guía operativa para liberaciones

Este documento resume los pasos necesarios para reproducir fallos, validar el
pipeline multi-stream y documentar los criterios de aceptación antes de liberar
una versión a producción.

## Reproducir fallos críticos

1. **Contrato de datos**. Ejecuta `python tools/ci_validate_data_contract.py`
   para generar un dataset sintético y verificar que los checks de `quality`
   coinciden con lo descrito en `docs/data_contract.md`.
2. **Entrenamiento de humo**. Corre `pytest tests/test_pipeline_end_to_end.py`
   para validar que el dataset sintético recorre el flujo completo
   (data → entrenamiento → exportación) utilizando `MultiStreamClassifier`.
3. **Exportación**. Usa `pytest tests/test_export.py` o
   `python tools/export_onnx_encoder_v9.py` sobre el checkpoint reportado en CI
   para confirmar que los artefactos TorchScript/ONNX son válidos.

## Criterios de aceptación

Un release candidato debe cumplir con los siguientes puntos:

- Los tests automáticos (`pytest`) deben pasar en CI y localmente sin
  modificaciones manuales.
- `python tools/ci_validate_metrics.py` debe informar que las pérdidas inicial y
  final del entrenamiento sintético están dentro de la tolerancia respecto a los
  valores documentados en el README.
- Los artefactos exportados con `tools/export_onnx_encoder_v9.py` deben superar
  las comprobaciones de `onnx.checker` y las comparaciones de TorchScript
  incluidas en `tests/test_pipeline_end_to_end.py`.

## Checklist previo a producción

- [ ] Generar un reporte de datos utilizando `tools/ci_validate_data_contract.py`
      y adjuntarlo en la bitácora de la release.
- [ ] Ejecutar la batería de tests (`pytest`) en un entorno limpio.
- [ ] Comparar las métricas del entrenamiento sintético con el README usando
      `tools/ci_validate_metrics.py`.
- [ ] Documentar cualquier desvío o ajuste en `docs/data_contract.md` y en la
      sección de métricas del README.
