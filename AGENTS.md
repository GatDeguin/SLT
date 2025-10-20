# Guía de contribución y documentación

Estas pautas aplican a todo el repositorio y complementan la información
operativa descrita en `README.md` y la carpeta `docs/`.

- Mantén un ancho máximo de 100 caracteres por línea en archivos de texto y
  Markdown.
- Prefiere títulos en estilo *sentence case* y listas descriptivas con verbos en
  infinitivo.
- Cada vez que modifiques la CLI o rutas por defecto, actualiza los ejemplos en
  `README.md` y enlaza las secciones relevantes dentro de `docs/`.
- Cuando describas flujos de datos, usa la convención de carpetas
  `data/single_signer/...` y el CSV `meta.csv` como referencia.
- Ejecuta, cuando sea viable, los controles de calidad locales: `pytest`,
  `ruff check .`, `black --check .` y `mypy`.
- Incluye en los mensajes de commit y en la documentación un resumen explícito
  de los componentes tocados (por ejemplo *dataset*, *entrenamiento*,
  *exportación*).
- Antes de abrir un PR confirma que cualquier cambio en los requerimientos se
  encuentra reflejado tanto en `requirements-dev.txt` como en la sección de
  instalación del README.
