# Checklist operativo para liberaciones

Esta guía resume los pasos necesarios para validar el pipeline multi-stream
antes de publicar un release. Úsala como complemento del README, la visión de
arquitectura (`docs/architecture_overview.md`) y las guías técnicas; cada ítem
enlaza con la documentación correspondiente.

> Nota: MediaPipe solo ofrece ruedas para Python <=3.12; usa esa versión al
> ejecutar utilidades dependientes de `tools/extract_rois_v2.py`.

## Reproducir errores y validar fixes

1. **Contrato de datos y keypoints**: ejecuta
   `python tools/ci_validate_data_contract.py` para generar un dataset sintético y
   confirmar que las máscaras por stream, los keypoints y las etiquetas
   CTC/gloss coinciden con `docs/data_contract.md`.
2. **Normalización de `meta.csv`**: lanza
   ```bash
   python tools/prepare_lsat_crops.py \
     --lsa-root data/single_signer/videos \
     --meta-csv meta.csv \
     --dry-run \
     --duration-threshold 20 \
     --delta-threshold 0.5 \
     --fail-on-outliers \
     --emit-split-json
   ```
   El resumen indica cuántos clips se descartaron, escribe `meta_missing.csv` y
   `meta_outliers.csv` junto al CSV original y genera `split_segments.jsonl` con
   los subtítulos parciales parseados. Revisa estos archivos antes de continuar.
3. **Alineación ROI-keypoints**: corre `pytest tests/data/test_lsa_t_multistream.py`
   para verificar que los keypoints por flujo siguen las máscaras y longitudes
   esperadas antes de lanzar experimentos reales.
4. **Entrenamiento de humo**: corre `pytest tests/test_pipeline_end_to_end.py`
   para comprobar que los componentes de datos, entrenamiento y exportación
   funcionan integrados. Si trabajas con flujos basados en keypoints, valida
   también `tools/train_slt_multistream_v9.py --use-mska` (o el wrapper) con un
   fragmento representativo.
5. **Evaluación**: valida `tools/eval_slt_multistream_v9.py` usando el checkpoint
   producido por la demo y revisa los reportes con
   `docs/metrics_dashboard_integration.py`.
6. **Exportación**: ejecuta `pytest tests/test_export.py` o el script
   `tools/export_onnx_encoder_v9.py` con el checkpoint de referencia para
   asegurar que ONNX y TorchScript son válidos. Complementa con
   `tools/test_realtime_pipeline.py` cuando existan regresiones de latencia.

## Criterios de aceptación

- Suite de `pytest` verde en un entorno limpio (sin caches previas).
- `ruff check .`, `black --check .` y `mypy` sin errores.
- `tools/ci_validate_metrics.py` reporta pérdidas dentro de la tolerancia
  documentada en el README.
- Los artefactos exportados superan `onnx.checker.check_model` y las comparaciones
  de TorchScript incluidas en los tests.
- Documentación actualizada con cualquier cambio en rutas, argumentos o
  dependencias.
- Los ejemplos de configuración (`configs/*.yml`, `docs/*.md`) reflejan los
  argumentos vigentes.

## Checklist previo a publicación

- [ ] Generar reporte de datos (`tools/ci_validate_data_contract.py`) y adjuntar
      el resultado en la bitácora de la release.
- [ ] Ejecutar `pytest tests/data/test_lsa_t_multistream.py` para confirmar la
      alineación ROI-keypoints y la propagación de glosas/CTC.
- [ ] Ejecutar `pytest` completo y conservar los registros.
- [ ] Comparar métricas con `tools/ci_validate_metrics.py` y documentar desvíos.
- [ ] Verificar que los ejemplos del README coinciden con la versión final del
      código.
- [ ] Confirmar que `requirements-dev.txt` y `pyproject.toml` reflejan cualquier
      dependencia nueva o actualizada.
- [ ] Validar que `tools/README.md` y `docs/pretraining.md` mencionan las nuevas
      banderas o scripts añadidos.
- [ ] Adjuntar capturas de los dashboards o reportes relevantes cuando cambien
      las métricas de referencia.
