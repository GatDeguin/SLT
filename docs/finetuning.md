# Fine-tuning multistream con inicialización MSKA

Esta guía resume cómo exportar los pesos auto-supervisados de rostro y manos, revisarlos y
aplicarlos como inicialización en `tools/train_slt_multistream_v9.py`.

## 1. Exportar los encoders especializados

1. Genera los crops siguiendo `docs/pretraining.md`.
2. Lanza los wrappers con `--export-checkpoint` y define un destino explícito:
   ```bash
   python tools/pretrain_dino_face.py \
     --train-dir data/single_signer/processed/face \
     --output-dir work_dirs/dino_face \
     --epochs 50 \
     --export-checkpoint \
     --output-path work_dirs/dino_face/encoder_face.pt

   python tools/pretrain_dino_hands.py \
     --train-dir data/single_signer/processed/hand_l \
     --output-dir work_dirs/dino_hands \
     --epochs 50 \
     --export-checkpoint \
     --output-path work_dirs/dino_hands/encoder_hand.pt
   ```
   El archivo se crea aunque exista `--export-backbone`; los metadatos registran el stream
   preentrenado, la pérdida mínima y la configuración del ViT.

## 2. Verificar los archivos exportados

Usa un script corto para inspeccionar el contenido y asegurar que `state_dict` esté presente:

```bash
python - <<'PY'
import torch
for path in [
    "work_dirs/dino_face/encoder_face.pt",
    "work_dirs/dino_hands/encoder_hand.pt",
]:
    payload = torch.load(path, map_location="cpu")
    print(path, payload.get("stream"), len(payload.get("state_dict", {})))
PY
```

El helper `slt.models.utils.load_mska_encoder_state` interpreta estos mapas y replica
automáticamente los pesos de manos sobre `hand_left` y `hand_right`.

## 3. Inicializar `train_slt_multistream_v9.py`

1. Ajusta el YAML base o usa `--set` para apuntar a los nuevos archivos:
   ```bash
   python tools/train_slt_multistream_v9.py \
     --config configs/single_signer_mska.yaml \
     --use-mska \
     --mska-face-state work_dirs/dino_face/encoder_face.pt \
     --mska-hand-state work_dirs/dino_hands/encoder_hand.pt \
     --face-backbone "file::work_dirs/dino_face/backbone.pt:dinov2_vits14" \
     --hand-left-backbone "file::work_dirs/dino_hands/backbone.pt:dinov2_vits14" \
     --hand-right-backbone "file::work_dirs/dino_hands/backbone.pt:dinov2_vits14"
   ```
2. El modelo invoca el helper durante la construcción de `MSKAEncoder`, por lo que no se requieren
   pasos adicionales. Los pesos se cargan antes de la integración con el encoder multistream y la
   etapa de MSKA queda lista para fine-tuning.

Consulta `docs/train_slt_multistream_v9.md` para detalles avanzados sobre MSKA y estrategias de
optimización combinando traducción, CTC y distillation.
