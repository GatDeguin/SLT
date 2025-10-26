# Preparar cache offline de modelos

Esta guía resume cómo descargar y exponer los artefactos de Hugging Face cuando
la máquina de entrenamiento o inferencia no dispone de conectividad externa.
Las instrucciones complementan la guía general de entrenamiento en
`docs/train_slt_multistream_v9.md` y las secciones de la CLI en el `README.md`.

## Descargar y organizar el tokenizer

1. Ejecuta `huggingface-cli download` (o `python -m transformers.models.auto`)
   desde una máquina con acceso a internet para el identificador requerido, por
   ejemplo:
   ```bash
   huggingface-cli download google/t5-v1_1-base --local-dir cache/tokenizer/t5
   ```
2. Copia la carpeta completa al entorno sin conexión y apunta la CLI a esa ruta
   mediante alguna de estas opciones:
   - Variable de entorno `SLT_TOKENIZER_PATH` o `SLT_TOKENIZER_DIR`.
   - Argumento `--tokenizer-search-path cache/tokenizer/t5` en
     `tools/train_slt_multistream_v9.py` o `tools/eval_slt_multistream_v9.py`.
   - Asigna directamente la ruta con `--tokenizer cache/tokenizer/t5`.
3. Añade `--tokenizer-local-files-only` para forzar que `create_tokenizer`
   omita cualquier solicitud de red y falle con un mensaje claro si la ruta no
   existe.

## Preparar pesos locales del decoder

1. Descarga el repositorio del modelo en una ubicación accesible sin conexión:
   ```bash
   huggingface-cli download facebook/mbart-large-cc25 \
     --local-dir cache/decoder/mbart-large-cc25
   ```
2. Expón la ruta usando la variable `SLT_DECODER_PATH` o el flag
   `--decoder-search-path cache/decoder/mbart-large-cc25` en la CLI de
   entrenamiento. Puedes enumerar varios directorios repitiendo el argumento.
3. Si prefieres que la CLI descargue artefactos antes de quedar offline, agrega
   `--decoder-hf-repo` junto a `--decoder-hf-filename pytorch_model.bin`.
   `TextSeq2SeqDecoder` invocará `hf_hub_download` y reutilizará la copia cacheada
   cuando `--decoder-local-files-only` esté activo.
4. Para rutas encapsuladas en variables de entorno crea `SLT_DECODER_DIR` con
   una lista separada por `os.pathsep` (`:` en Unix, `;` en Windows). El helper
   buscará entradas válidas en orden.

## Ejecutar las CLIs en modo offline

- **Entrenamiento** (`tools/train_slt_multistream_v9.py`): combina
  `--tokenizer-local-files-only --decoder-local-files-only` con rutas explícitas.
  Ejemplo:
  ```bash
  python tools/train_slt_multistream_v9.py \
    --decoder-preset mska_paper_mbart \
    --tokenizer-search-path cache/tokenizer/t5 \
    --decoder-search-path cache/decoder/mbart-large-cc25 \
    --tokenizer-local-files-only --decoder-local-files-only \
    ...
  ```
  Cualquier preset aplica las rutas adicionales sobre `ModelConfig`, por lo que
  no es necesario editar el YAML.
- **Evaluación** (`tools/eval_slt_multistream_v9.py`): exige un checkpoint local
  (`--checkpoint`). Añade `--tokenizer-search-path` y
  `--tokenizer-local-files-only` para reutilizar el cache del tokenizer durante
  la decodificación autoregresiva.

En ambos casos `create_tokenizer` y `TextSeq2SeqDecoder` muestran mensajes
explícitos cuando no encuentran las rutas declaradas. Aprovecha `SLT_TOKENIZER_*`
y `SLT_DECODER_*` en tus scripts de orquestación para evitar repetir rutas en
cada ejecución.
