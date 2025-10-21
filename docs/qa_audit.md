# Qa audit summary

## Entorno de pruebas

- Ejecutar `python -m pip install -e .` para instalar el paquete en modo editable.
- Instalar dependencias opcionales `onnx`, `onnxruntime` y `onnxscript` para los
  casos de exportación.

## Resultados de pruebas automatizadas

- `pytest` sin dependencias de exportación adicionales: 86 tests aprobados, 4
  omitidos por falta de `onnx`/`onnxruntime`/CUDA. 【876514†L1-L70】
- `pytest` con dependencias opcionales: fallos en exportación ONNX y pipeline
  extremo a extremo. 【6413cb†L1-L88】
- `ruff check .`: 767 infracciones (mayoría de estilo y migración a anotaciones
  modernas). 【88b53f†L1-L110】
- `black --check .`: 37 archivos requieren formateo. 【426c0e†L1-L9】
- `mypy slt tools`: 2 errores por tipados ausentes y path duplicado. 【f07884†L1-L9】

## Observaciones clave

- Los tests de exportación fallan porque `_select_state_dict` no reconoce la
  estructura de pesos generada por `MultiStreamEncoder` con backbones ViT, al
  esperar la variante `_TinyBackbone`. 【22d440†L49-L106】【dc67cb†L222-L260】
- Las pruebas end-to-end fallan al invocar `MultiStreamClassifier.forward` con
  un argumento `inputs` anidado; `_split_batch` no descarta `video_ids` y
  reenvía el diccionario completo. 【376d07†L44-L111】【a1550d†L6-L41】
- La integración de backends en tiempo real se omite por falta del runtime de
  OpenCV (`libGL.so.1` ausente). 【5b5886†L1-L9】【0d4a57†L1-L12】
- Numerosas advertencias en los tests señalan dependencias opcionales ausentes
  (xFormers, GradScaler CUDA) y deprecaciones que deberían atenderse. 【6413cb†L89-L118】

## Recomendaciones

1. Unificar la estrategia de backbones entre entrenamiento y exportación para
   que `_select_state_dict` pueda mapear pesos ViT a `_TinyBackbone`, o ajustar
   las pruebas para reflejar la configuración soportada. 【bf81d4†L1-L12】
2. Actualizar `_split_batch` para eliminar metadatos no consumidos antes de
   llamar al modelo o adaptar `build_collate` para devolver objetivos en una
   clave dedicada. 【376d07†L98-L140】【a1550d†L6-L41】
3. Documentar dependencias del sistema para OpenCV (p. ej. `libgl1` en Debian)
   y contemplar un backend alternativo en entornos headless. 【5b5886†L1-L9】
4. Planificar una limpieza de estilo siguiendo `ruff`/`black` y añadir stubs de
   `PyYAML` para pasar `mypy`. 【88b53f†L1-L110】【c6006f†L1-L9】【f07884†L1-L9】
