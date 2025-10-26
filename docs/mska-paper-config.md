# MSKA-SLT hyper-parameters

El paper *Multi-Stream Keypoint Attention Network for Sign Language Recognition and Translation*
resume la arquitectura MSKA y el entrenamiento de la rama de traducción con mBART.

> "The network employs four streams, with each stream consisting 8 attention blocks, and each block
> containing 6 attention heads. The output channels are set as follows: 64 ,64,128,128,256,256,256
> and 256 respectively."

> "We initialize our translation network with mBART-large-cc25 pretrained on CC25. We use a beam
> width of 5 for both the CTC decoder and the SLT decoder during inference. We train for 40 epochs
> with an initial learning rate of 1e−3 for the MLP and 1e−5 for MSKA-SLR and the translation
> network in MSKA-SLT."

> "The whole network is jointly supervised by the CTC losses and the self-distillation losses ..."
> "Table 4d shows that our MSKA-SLR attains the best performance when the weight is set to 1.0."

El preset `mska_paper_mbart` lleva estos parámetros al YAML:

- Establece `projector_dim=64` y `d_model=256` para escalar la fusión multi-stream a las ocho
  etapas descritas en el paper (64→256). Los bloques temporales y las cabezas MSKA se fijan en 8 y
  6 respectivamente.
- Activa las pérdidas auxiliares con `mska_ctc_weight=1.0` y `mska_distillation_weight=1.0`,
  manteniendo la temperatura de distilación en `1.0` como en la tabla 4d.
- expone variantes de decoder: mBART-large CC25 como opción por defecto para inferencia offline y
  T5 v1.1 Base como alternativa liviana. Ambas se seleccionan automáticamente cuando se indica
  `--decoder-model mbart` o `--decoder-model t5` desde la CLI.

Los campos pueden ajustarse con `--mska-*` y `--decoder-*` al invocar
`tools/train_slt_multistream_v9.py` para replicar ablations o experimentos adicionales.
