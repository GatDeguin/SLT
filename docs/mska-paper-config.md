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

Este preset replica dichos hiperparámetros en la configuración por defecto y expone banderas CLI
para ajustar el número de cabezas, bloques temporales y pesos de pérdida cuando sea necesario.
