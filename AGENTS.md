# Guía para contribuidores

Estas pautas aplican a todo el repositorio.

- Mantén un ancho máximo de 100 caracteres por línea en archivos de texto.
- Prefiere títulos en estilo *sentence case* en Markdown.
- Antes de abrir un PR, sincroniza la documentación y ejemplos de CLI cuando cambies rutas,
  argumentos o salidas relevantes.
- Verifica que los comandos de referencia utilicen rutas válidas dentro del dataset
  `single_signer` y el CSV `meta.csv`.
- Ejecuta los siguientes comandos de control de calidad cuando sea posible:
  - `pytest`
  - `ruff check .`
  - `black --check .`
  - `mypy`
- Añade enlaces cruzados entre `README.md` y los documentos técnicos cuando agregues nuevas
  guías.
