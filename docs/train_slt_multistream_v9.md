# Guía de entrenamiento `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el encoder multi-stream y
un decoder seq2seq compatible con HuggingFace utilizando los datos preparados en
`data/single_signer/processed/` y los índices derivados de `meta.csv`. De forma
predeterminada el modelo se inicializa con el checkpoint validado
`single_signer` cuando el archivo descargado está disponible (en
`data/single_signer/` o vía `SLT_SINGLE_SIGNER_CHECKPOINT`), aunque puedes
deshabilitarlo con `--pretrained none`. Esta guía resume los argumentos más
importantes, patrones de uso avanzados y recomendaciones para reproducir
experimentos. Complementa la referencia rápida incluida en `tools/README.md`.

## Requisitos previos

- Streams generados con `tools/extract_rois_v2.py` siguiendo el contrato de datos.
- CSV `meta.csv` con columnas `video_id;texto` y splits `train.csv`/`val.csv`/`test.csv`.
- Tokenizador HuggingFace compatible con modelos tipo T5/BART.

## Pipeline unificado ROI + keypoints

1. **Preprocesamiento**: extrae las ROIs con `tools/extract_rois_v2.py`, genera
   los keypoints MediaPipe por video y prepara `gloss.csv` con las secuencias y
   etiquetas CTC. `docs/data_contract.md` detalla formatos y convenciones.
2. **Validación**: ejecuta `python tools/ci_validate_data_contract.py` o las
   pruebas de `tests/data/test_lsa_t_multistream.py` para confirmar que las
   máscaras por stream y la alineación ROI-keypoints son coherentes.
3. **Entrenamiento multi-pérdida**: lanza este script con `--use-mska` y define
   `--mska-translation-weight`, `--mska-ctc-weight` y
   `--mska-distillation-weight` según el objetivo del experimento. El modelo
   combina automáticamente las pérdidas activas y registra cada término en
   `metrics.jsonl`. Cuando MSKA está activado, la representación fusionada de
   keypoints se proyecta mediante un MLP con hasta dos capas ocultas
   (`Linear → LeakyReLU → Dropout opcional → Linear → LeakyReLU → Dropout
   opcional → Linear`) controladas por `--mska-gloss-hidden-dim` y
   `--mska-gloss-second-hidden-dim`; el coeficiente de fuga (0.01 por defecto)
   se mantiene compartido y la secuencia resultante se expone al decoder como
   glosas para aplicar la combinación LSLT de traducción y reconocimiento.
4. **Evaluación y exportación**: reutiliza las mismas banderas MSKA en
   `tools/eval_slt_multistream_v9.py` y `tools/export_onnx_encoder_v9.py` para
   mantener consistencia entre entrenamiento, evaluación y despliegue.

## Stream gating regularizer (SGR)

El SGR añade a cada stream de keypoints una matriz de atención global que se
combina con la auto-atención local tanh del encoder MSKA. La matriz se inicializa
como identidad y se amplía de forma perezosa cuando aumenta la secuencia,
manteniendo los pesos previos y adaptando automáticamente dispositivo y tipo de
dato. 【F:slt/models/mska.py†L33-L62】 Durante el *forward* la matriz activada se
normaliza, se filtra con las máscaras válidas y se interpola con la atención
local mediante `global_mix` (0.0 usa solo la atención local, 1.0 emplea la matriz
global normalizada). 【F:slt/models/mska.py†L193-L211】

### Activaciones disponibles

`--mska-sgr-activation` admite `softmax`, `sigmoid`, `tanh`, `relu` e
`identity`/`linear`/`none`. Las funciones se aplican elemento a elemento sobre la
matriz antes de normalizarla y descartan valores negativos tras la activación,
lo que permite controlar si la matriz actúa como un kernel probabilístico o
como un mapa de calor con soporte esparso. 【F:slt/models/mska.py†L121-L133】

### Compartir o especializar la matriz

Al activar `--mska-sgr-shared` se crea un único `_GlobalAttentionStore` que
sirve a todos los streams MSKA, garantizando que la matriz aprendida sea común y
se reutilice en cada paso. 【F:slt/models/mska.py†L806-L832】 Esta opción estabiliza
experimentos con pocos ejemplos por stream (p. ej. rostro) y reduce el número de
parámetros adicionales.

Si usas `--mska-sgr-per-stream` cada `KeypointStreamEncoder` mantiene su propio
almacén y los gradientes solo afectan a la matriz de dicho flujo, lo que resulta
útil cuando los streams capturan dinámicas muy distintas (pose vs. manos) o
cuando buscas interpretar qué articulaciones dominan en cada modalidad.

### Seguimiento de métricas

El impacto del SGR se refleja en las métricas que ya escribe `metrics.jsonl`:
`loss_translation_weighted`, `loss_ctc_weighted`, `loss_distillation_weighted` y
`perplexity` permiten comparar ejecuciones con y sin SGR. Tras cada época
puedes inspeccionarlas con `jq` para revisar tendencias:

```bash
jq '{epoch, loss_translation_weighted, loss_ctc_weighted, loss_distillation_weighted, perplexity}' \
  work_dirs/experimento_sgr/metrics.jsonl | tail
```

Una caída sostenida en las pérdidas auxiliares indica que la matriz global está
favoreciendo un alineamiento más consistente entre streams; si las métricas se
mueven en sentido contrario conviene probar otra activación o reducir
`--mska-sgr-mix` para priorizar la atención local.

## Ejecución básica

```bash
python tools/train_slt_multistream_v9.py \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --work-dir work_dirs/single_signer_experiment \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --epochs 40 --batch-size 4 --sequence-length 64
```

Artefactos generados en `--work-dir`:

- `last.pt`: checkpoint de la última época.
- `best.pt`: mejor pérdida de validación observada.
- `config.json`: configuración efectiva combinando defaults, archivo y CLI.
- `metrics.jsonl`: historial con pérdidas y métricas por época.

## Fine-tuning con T5 v1.1 Base (`--decoder-preset signmusketeers`)

El preset `signmusketeers` replica la configuración descrita en
`configs/presets/decoder_signmusketeers_t5.yaml`, ajustando el espacio latente a 768
dimensiones y concatenando las corrientes de rostro, manos y pose antes de un decoder
`google/t5-v1_1-base`. 【F:configs/presets/decoder_signmusketeers_t5.yaml†L1-L31】

Pasos recomendados:

1. **Verifica dependencias**: instala `transformers>=4.40` y asegúrate de haber ejecutado
   `huggingface-cli login` si el checkpoint requiere autenticación.
2. **Lanza el entrenamiento** usando el preset. El tokenizador se resuelve automáticamente a
   `google/t5-v1_1-base`, por lo que no es necesario especificar `--tokenizer`.
   ```bash
   python tools/train_slt_multistream_v9.py \
     --decoder-preset signmusketeers \
     --face-dir data/single_signer/processed/face \
     --hand-left-dir data/single_signer/processed/hand_l \
     --hand-right-dir data/single_signer/processed/hand_r \
     --pose-dir data/single_signer/processed/pose \
     --metadata-csv meta.csv \
     --train-index data/single_signer/index/train.csv \
     --val-index data/single_signer/index/val.csv \
     --work-dir work_dirs/signmusketeers_t5 \
     --batch-size 4 --sequence-length 128
   ```
3. **Monitorea recursos**: con `batch-size` 4 y `sequence-length` 128 el uso de memoria
   ronda los 22 GB en GPUs tipo RTX 3090/4090. Reduce el lote o el tamaño de secuencia si
   trabajas con 16 GB.
4. **Evalúa la convergencia**: tras ~30 épocas deberías observar pérdidas de validación
   entre 2.3 y 2.6 y CER en torno a 0.55 en `data/single_signer`. Ajusta `--lr` o aplica
   acumulación de gradientes si no se estabiliza.

El preset está inspirado en el paper SignMusketeers, que reporta ~14 BLEU en How2Sign con
esta arquitectura multi-stream. 【F:docs/signmusketeers-paper-summary.md†L9-L33】

## Traducción offline con mBART (`--decoder-preset mska_paper_mbart`)

La configuración `mska_paper_mbart` replica los 8 bloques atencionales con 6 cabezas descritos en el
paper MSKA-SLT y selecciona el decoder `facebook/mbart-large-cc25` para ejecutar inferencia sin
dependencia de servicios externos. 【F:configs/presets/mska_paper_mbart.yaml†L1-L47】

```bash
python tools/train_slt_multistream_v9.py \
  --decoder-preset mska_paper_mbart \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --keypoints-dir data/single_signer/processed/keypoints \
  --gloss-csv data/single_signer/gloss.csv \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --work-dir work_dirs/mska_mbart_offline \
  --batch-size 4 --sequence-length 128
```

El preset fija `lr=1e-5`, `weight_decay=1e-3`, `epochs=40` y preserva los pesos MSKA
(`mska_ctc_weight` e `mska_distillation_weight`) en 1.0 conforme al paper original.
【F:configs/presets/mska_paper_mbart.yaml†L24-L47】【F:docs/mska-paper-config.md†L1-L17】
Los flags `--mska-heads`, `--mska-stream-heads` y `--mska-temporal-blocks` pueden ajustar la
configuración sin editar el YAML; el parser reescribe el campo correspondiente de `ModelConfig` tras
cargar el preset. Si necesitas intercambiar el decoder sin abandonar los valores MSKA, usa
`--decoder-model mbart` (o `mbart-large`) para normalizar automáticamente la ruta a
`facebook/mbart-large-cc25`.

## Prompt tuning del decoder T5

Cuando el decoder es un T5 (`model_type=t5` o presets basados en T5) puedes inyectar
embeddings de prompt aprendibles delante de las salidas del encoder. Esto permite modular
la atención cruzada y acelerar la adaptación a nuevos dominios sin tocar el vocabulario.

- `--decoder-prompt-length N`: activa `N` embeddings aprendibles (por defecto `0`, sin
  prompt).
- `--decoder-prompt-init {normal,zero,uniform,tokens,vocab}`: controla la inicialización.
  `normal` usa desviación 0.02, `tokens` copia embeddings existentes y `vocab` muestrea
  filas del embedding compartido.
- `--decoder-prompt-text` o `--decoder-prompt-tokens`: inicializan el prompt a partir de
  un texto o de IDs explícitos; si defines tokens y no especificas la longitud, se usará
  automáticamente el número de tokens.
- `--decoder-prompt-std` y `--decoder-prompt-range`: ajustan la escala para las variantes
  `normal` y `uniform` respectivamente.

Ejemplo con un prompt de 20 tokens inicializado con la frase "translate to rioplatense":

```bash
python tools/train_slt_multistream_v9.py \
  --decoder-preset signmusketeers \
  --decoder-prompt-length 20 \
  --decoder-prompt-init tokens \
  --decoder-prompt-text "translate to rioplatense" \
  --teacher-forcing-mode scheduled \
  --teacher-forcing-ratio 1.0 \
  --teacher-forcing-min-ratio 0.3 \
  --teacher-forcing-decay 0.9 \
  ...
```

En validaciones internas un prompt de 10-32 tokens con `prompt-init=tokens` y una frase
semánticamente cercana al dominio objetivo redujo la pérdida de validación en ~0.08 y el
CER en ~1.5 puntos tras 15 épocas. Valores mayores a 48 tokens no aportaron mejoras y
añadieron ~5 % de uso de memoria. Si no cuentas con un texto representativo, usa
`--decoder-prompt-init vocab` para muestrear el embedding compartido.

## Teacher forcing y scheduled sampling

Además del teacher forcing estándar (probabilidad 1.0 de alimentar el token objetivo), el
script permite aplicar scheduled sampling para exponer gradualmente el decoder a sus
propias predicciones:

- `--teacher-forcing-mode {standard,scheduled}`: activa la modalidad deseada.
- `--teacher-forcing-ratio`: probabilidad inicial de usar el token de referencia.
- `--teacher-forcing-min-ratio`: cota inferior aplicada cuando la probabilidad decae.
- `--teacher-forcing-decay`: factor multiplicativo por época (ej. `0.9` reduce 10 % cada
  ciclo hasta alcanzar la cota mínima).

Durante el entrenamiento la consola y `metrics.jsonl` registran el ratio efectivo aplicado
en cada época (`record.teacher_forcing`). En pruebas sobre `single_signer` con prompts
activos, un schedule `ratio=1.0`, `min_ratio=0.4`, `decay=0.92` redujo el CER validación
0.7 puntos y mitigó explosiones de pérdida en épocas tardías.

## Argumentos destacados

### Datos

| Bandera | Descripción |
|---------|-------------|
| `--face-dir`, `--hand-left-dir`, `--hand-right-dir` | Directorios de recortes RGB. |
| `--pose-dir` | Directorio con los `.npz` de pose. |
| `--metadata-csv` | Ruta a `meta.csv`. |
| `--keypoints-dir` | Directorio opcional con keypoints MediaPipe (.npy/.npz). |
| `--gloss-csv` | CSV con columnas `video_id;gloss;ctc_labels` para pérdidas MSKA. |
| `--train-index`, `--val-index` | Listas de `video_id` para cada split. |
| `--batch-size`, `--val-batch-size` | Tamaños de lote de entrenamiento y validación. |
| `--num-workers`, `--pin-memory` | Parámetros del `DataLoader`. |
| `--mix-stream STREAM[:P]` | Permuta streams individuales con probabilidad `P`. |
| `--max-target-length` | Longitud máxima de la secuencia tokenizada. |

### Augmentaciones de keypoints

- `--keypoint-normalize-center` / `--no-keypoint-normalize-center`: controla si los
  keypoints se desplazan al centro de la imagen antes de aplicar las
  transformaciones. La normalización está habilitada por defecto.
- `--keypoint-scale-range min,max`: rango uniforme (``>0``) para escalar los
  keypoints alrededor del centro.
- `--keypoint-translate-range`: traslaciones en el plano. Admite un valor (±X),
  dos valores (`min,max`) o cuatro valores (`min_x,max_x,min_y,max_y`).
- `--keypoint-rotate-range`: ángulos mínimo y máximo, en grados, para rotar los
  keypoints alrededor del centro.
- `--keypoint-resample-range`: factores (``>0``) utilizados para re-muestrear la
  secuencia temporal de keypoints antes del muestreo final a `T` frames.

Ejemplo con augmentaciones activas durante el entrenamiento:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --keypoint-scale-range 0.9,1.1 \
  --keypoint-translate-range -0.05,0.05 \
  --keypoint-rotate-range -15,15 \
  --keypoint-resample-range 0.85,1.1
```

Los mismos flags están disponibles en `tools/eval_slt_multistream_v9.py` para
replicar el preprocesamiento al evaluar checkpoints.

### Modelo

| Bandera | Descripción |
|---------|-------------|
| `--image-size`, `--sequence-length` | Resolución espacial y temporal del encoder. |
| `--projector-dim`, `--d-model` | Dimensiones internas del encoder. |
| `--pose-landmarks` | Cantidad de puntos de pose (se multiplica por 3 ejes). |
| `--temporal-*` | Configuración del transformer temporal. |
| `--decoder-layers`, `--decoder-heads`, `--decoder-dropout` | Arquitectura del decoder. |
| `--decoder-model`, `--decoder-config` | Modelos/configs HuggingFace precargados. |
| `--decoder-class`, `--decoder-kwargs` | Decoder Python personalizado vía módulo/clase. |
| `--pretrained` | Selecciona `single_signer` (default) o `none` para inicializar pesos. |
| `--pretrained-checkpoint` | Ruta al checkpoint `single_signer` descargado. |
| `--use-mska` | Activa la rama MSKA (requiere keypoints y glosas). |
| `--decoder-prompt-*` | Activa e inicializa el prompt aprendible del decoder T5. |
| `--mska-heads` | Número de cabezas en la atención multi-stream. |
| `--mska-ff-multiplier` | Factor del bloque feed-forward dentro de la atención. |
| `--mska-dropout` | Dropout aplicado a proyectores y cabezas MSKA. |
| `--leaky-relu-negative-slope` | Coeficiente de fuga de LeakyReLU en MSKA (default: 0.01). |
| `--mska-input-dim` | Dimensionalidad de los keypoints de entrada. |
| `--mska-use-sgr` / `--mska-no-sgr` | Activa o desactiva la matriz global compartida (SGR). |
| `--mska-sgr-activation` | Activación SGR (`softmax`/`sigmoid`/`tanh`/`relu`/`identity`). |
| `--mska-sgr-mix` | Mezcla entre la atención local y la matriz SGR (0.0 a 1.0). |
| `--mska-sgr-shared` | Usa una matriz global compartida. |
| `--mska-sgr-per-stream` | Aprende una matriz por stream. |
| `--mska-ctc-vocab` | Tamaño del vocabulario para las cabezas CTC auxiliares. |
| `--mska-detach-teacher` | Controla si los logits fusionados participan en la distilación. |
| `--mska-stream-heads` | Número de cabezas en la atención por articulación. |
| `--mska-temporal-blocks` | Cantidad de bloques convolucionales temporales por stream. |
| `--mska-temporal-kernel` | Tamaño del kernel temporal usado en dichos bloques. |
| `--mska-temporal-dilation` | Dilatación temporal aplicada de forma compartida en los bloques. |
| `--mska-translation-weight` | Peso de la pérdida de traducción. |
| `--mska-ctc-weight` | Peso del término CTC auxiliar. |
| `--mska-distillation-weight` | Peso del término de distilación. |
| `--mska-distillation-temperature` | Temperatura aplicada al término de distilación. |
| `--mska-gloss-hidden-dim` | Dimensión oculta del MLP que proyecta la secuencia MSKA. |
| `--mska-gloss-second-hidden-dim` | Segunda capa oculta del MLP de glosas. |
| `--mska-gloss-activation` | Activación del MLP de glosas (`leaky_relu`). |
| `--mska-gloss-dropout` | Dropout aplicado entre las capas del MLP de glosas. |
| `--mska-gloss-fusion` | Fusión de glosas con el decoder (`add`/`concat`/`none`). |

### Optimización y ejecución

| Bandera | Descripción |
|---------|-------------|
| `--optimizer`, `--lr`, `--weight-decay` | Hiperparámetros del optimizador (`AdamW` por defecto). |
| `--scheduler` + parámetros | `none`, `steplr` o `cosine` con banderas asociadas. |
| `--label-smoothing` | Factor aplicado a la pérdida de entropía cruzada. |
| `--precision {amp,float32}` | Controla AMP automático en GPU. |
| `--init-checkpoint` | Inicializa pesos desde un checkpoint previo. |
| `--resume` | Restaura entrenamiento completo (modelo, optimizador, scaler). |
| `--seed`, `--device` | Control determinista y selección de dispositivo. |
| `--teacher-forcing-*` | Controla el ratio de teacher forcing y scheduled sampling. |

## Plantillas de configuración (`--config`)

El script acepta archivos JSON/YAML con secciones `data`, `model`, `optim` y
`training`. Ejemplo mínimo:

```yaml
data:
  face_dir: data/single_signer/processed/face
  hand_left_dir: data/single_signer/processed/hand_l
  hand_right_dir: data/single_signer/processed/hand_r
  pose_dir: data/single_signer/processed/pose
  metadata_csv: meta.csv
  train_index: data/single_signer/index/train.csv
  val_index: data/single_signer/index/val.csv
  work_dir: work_dirs/single_signer_experiment
model:
  sequence_length: 64
  image_size: 224
training:
  epochs: 40
optim:
  lr: 0.0005
```

Ejemplo de MLP MSKA con dos capas ocultas y dropout:

```yaml
model:
  use_mska: true
  mska_gloss_hidden_dim: 256
  mska_gloss_second_hidden_dim: 128
  mska_gloss_dropout: 0.1
```

Sobrescribe campos puntuales usando `--set clave=valor`, por ejemplo:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set data.batch_size=6 \
  --set optim.scheduler="cosine" \
  --set model.decoder_heads=6
```

Los valores efectivos se persistirán en `config.json` dentro de `work_dir`.

### Gestión avanzada de configuraciones

- Usa notación de puntos para campos anidados (`data.batch_size=6`).
- Valores booleanos aceptan `true`/`false` (insensible a mayúsculas).
- Strings con espacios deben ir entre comillas (`"cosine warmup"`).
- Repite `--set` tantas veces como sea necesario; la última definición prevalece.

Guarda los archivos YAML/JSON utilizados junto al `work_dir` para asegurar
reproducibilidad. Los diffs de configuración se listan en `config.json` bajo la
clave `overrides`.

## Decodificadores personalizados

Para usar un decoder propio, proporciona `--decoder-class` con el formato
`paquete.modulo:Clase` y argumentos opcionales en JSON:

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --decoder-class my_project.decoders:MyDecoder \
  --decoder-kwargs '{"dropout": 0.0, "share_embeddings": true}'
```

Verifica que la dimensionalidad (`d_model`) del decoder coincida con el encoder
multi-stream. El script valida este punto al inicializar el modelo.

### Mezcla de streams y augmentations

- `--mix-stream stream[:prob]`: intercambia aleatoriamente streams entre clips
  durante el entrenamiento. Omite `:prob` para usar el valor por defecto `1.0`.
  Los nombres válidos son `face`, `hand-left`, `hand-right` y `pose`.
- En archivos `--config` puedes definir `mix_streams` como diccionario:
  ```yaml
  data:
    mix_streams:
      face: 0.5
      hand_left: 0.25
  ```
  Las claves usan guion bajo para coincidencia con los campos del dataset.

`normalise_mix_spec` garantiza que las probabilidades se ajusten al rango `[0,1]`
y emite errores descriptivos cuando la configuración es inválida.

### Optimización avanzada

- `--grad-accum-steps`: acumula gradientes para simular lotes más grandes.
- `--clip-grad-norm`: aplica *gradient clipping* antes de cada actualización.
- `--compile` / `--no-compile`: activa `torch.compile` con el modo indicado en
  `--compile-mode` cuando la versión de PyTorch lo soporta.
- `--precision amp`: habilita *automatic mixed precision* en GPU.

## Reanudaciones y warm start

- `--resume`: requiere que `--work-dir` contenga `last.pt`. Restaura optimizador,
  scaler de AMP y contadores internos.
- `--init-checkpoint`: carga solo los pesos del modelo antes de comenzar la
  primera época. Útil para transferir pesos desde otra sesión.

## Métricas y seguimiento

- El archivo `metrics.jsonl` guarda un objeto JSON por época con pérdidas de
  entrenamiento y validación.
- Usa `tools/ci_validate_metrics.py` para comparar las métricas contra valores de
  referencia (se ejecuta en CI).
- La bandera `--tensorboard` habilita logging directo compatible con
  `tensorboard --logdir work_dirs/...`.
- `work_dir/logs/` almacena los archivos de texto con métricas resumidas por
  época y mensajes de `logging`.

Los reportes incluyen información de tiempo por iteración, tasa de samples y
distribución de pérdidas. Conserva estos artefactos para facilitar auditorías y
reproducibilidad.

## Siguientes pasos

1. Evalúa los checkpoints con `tools/eval_slt_multistream_v9.py` siguiendo el
   flujo descrito en el README.
2. Exporta el encoder con `tools/export_onnx_encoder_v9.py` para integrarlo en
   demos en tiempo real o servicios de inferencia.
3. Si necesitas sustituir backbones, revisa `docs/pretraining.md` para cargar
   pesos provenientes de DINO/iBOT mediante `load_dinov2_backbone`.
