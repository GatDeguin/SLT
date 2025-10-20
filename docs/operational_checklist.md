# Checklist operativo para liberaciones

Esta guía resume los pasos necesarios para validar el pipeline multi-stream antes
de publicar un release. Úsala como complemento del README y las guías técnicas.

## Reproducir errores y validar fixes

1. **Contrato de datos**: ejecuta `python tools/ci_validate_data_contract.py` para
   generar un dataset sintético y confirmar que los checks de calidad coinciden
   con `docs/data_contract.md`.
2. **Entrenamiento de humo**: corre `pytest tests/test_pipeline_end_to_end.py`
   para comprobar que los componentes de datos, entrenamiento y exportación
   funcionan integrados.
3. **Evaluación**: valida `tools/eval_slt_multistream_v9.py` usando el checkpoint
   producido por la demo y revisa los reportes con
   `docs/metrics_dashboard_integration.py`.
4. **Exportación**: ejecuta `pytest tests/test_export.py` o el script
   `tools/export_onnx_encoder_v9.py` con el checkpoint de referencia para
   asegurar que ONNX y TorchScript son válidos.

## Criterios de aceptación

- Suite de `pytest` verde en un entorno limpio (sin caches previas).
- `ruff check .`, `black --check .` y `mypy` sin errores.
- `tools/ci_validate_metrics.py` reporta pérdidas dentro de la tolerancia
  documentada en el README.
- Los artefactos exportados superan `onnx.checker.check_model` y las comparaciones
  de TorchScript incluidas en los tests.
- Documentación actualizada con cualquier cambio en rutas, argumentos o
  dependencias.

## Checklist previo a publicación

- [ ] Generar reporte de datos (`tools/ci_validate_data_contract.py`) y adjuntar
      el resultado en la bitácora de la release.
- [ ] Ejecutar `pytest` completo y conservar los registros.
- [ ] Comparar métricas con `tools/ci_validate_metrics.py` y documentar desvíos.
- [ ] Verificar que los ejemplos del README coinciden con la versión final del
      código.
- [ ] Confirmar que `requirements-dev.txt` y `pyproject.toml` reflejan cualquier
      dependencia nueva o actualizada.
