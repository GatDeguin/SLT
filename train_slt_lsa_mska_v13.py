# -*- coding: utf-8 -*-
"""
Entrenamiento SLT (LSA) desde keypoints 79 (XY o XYC) + mBART-50 como decodificador.
Robusto a .npy en formatos: (T,79,3), (T,79,2), (T,237), (T,158).

Uso típico:
python tools/train_slt_lsa_mska_v13.py \
  --kp_dir "G:\...\lsat_kp" \
  --csv    "G:\...\meta.csv" --csv_delim ";" \
  --work_dir "G:\...\work_dirs\slt_lsa" \
  --device cuda \
  --epochs 50 --batch_size 4 --num_workers 0 \
  --xy_only true --min_conf 0.25 \
  --nhead 8 --nlayers 8 --d_model 1024 \
  --lr_encoder 3e-4 --lr_decoder 1e-5 --t_max 40 \
  --tok_name "facebook/mbart-large-50-many-to-many-mmt" \
  --tgt_lang es_XX --unfreeze_last_n_dec_layers 2 \
  --beam 5 --max_gen_len 64

"""

import os, re, math, json, argparse, random, warnings
import os.path as osp
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizerFast, MBartTokenizer
from transformers.modeling_outputs import BaseModelOutput

try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_OK = True
except Exception:
    SACREBLEU_OK = False
    warnings.warn("sacrebleu >=2 no disponible. BLEU reportado como 0.0.")

# ---------------------------
# Utilidades
# ---------------------------

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def plog(s: str):
    print(s, flush=True)

def try_read_csv(path: str, sep: str):
    """Carga CSV tolerante a codificaciones y líneas problemáticas."""
    encs = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    last_err = None
    for e in encs:
        try:
            return pd.read_csv(path, sep=sep, encoding=e, on_bad_lines='skip')
        except Exception as ex:
            last_err = ex
    raise last_err

def list_npy(kp_dir: str) -> Dict[str, str]:
    paths = []
    for root, _, files in os.walk(kp_dir):
        for f in files:
            if f.lower().endswith('.npy'):
                paths.append(osp.join(root, f))
    bases = {osp.splitext(osp.basename(p))[0]: p for p in sorted(paths)}
    return bases

def parse_time_any(x) -> Optional[float]:
    """
    Convierte un 'start'/'end' variopinto a segundos (float).
    Acepta:
      - float/int directo
      - "mm:ss(.fff)" o "hh:mm:ss(.fff)"
      - "2.996.877" -> se normaliza quitando puntos repetidos; si queda entero grande, se devuelve tal cual (segundos)
      - "2,5" -> 2.5
    Devuelve None si no es interpretable.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None

    # hh:mm:ss(.ms) | mm:ss(.ms)
    if ':' in s:
        try:
            parts = s.split(':')
            parts = [p.strip() for p in parts]
            parts = [p.replace(',', '.') for p in parts]
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                mm, ss = parts
                return 60.0 * mm + ss
            elif len(parts) == 3:
                hh, mm, ss = parts
                return 3600.0 * hh + 60.0 * mm + ss
        except Exception:
            pass

    # reemplazar ',' decimal por '.'
    s2 = s.replace(',', '.')
    # si hay más de un '.', probablemente separadores de miles mal puestos.
    if s2.count('.') > 1:
        # conservar solo el último '.' como decimal; quitar los anteriores
        left, dot, right = s2.rpartition('.')
        left = left.replace('.', '')
        s2 = left + '.' + right

    try:
        return float(s2)
    except Exception:
        return None

# ---------------------------
# Normalización de KP
# ---------------------------

def ensure_xyc(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Devuelve (xyz, has_conf) con forma:
      - Si input es (T,79,3) -> (T,79,3), has_conf=True
      - Si input es (T,79,2) -> (T,79,3) con c=1, has_conf=False
      - Si input es (T,237)  -> (T,79,3)
      - Si input es (T,158)  -> (T,79,3) con c=1
    Cualquier otra forma se intenta inferir; si no es posible, lanza ValueError.
    """
    if arr.ndim == 3:
        T, J, D = arr.shape
        if J == 0 or T == 0:
            raise ValueError("Clip vacío o sin joints.")
        if D == 3:
            xyz = arr.astype(np.float32)
            return xyz, True
        elif D == 2:
            xy = arr.astype(np.float32)
            c = np.ones((T, J, 1), dtype=np.float32)
            xyz = np.concatenate([xy, c], axis=2)
            return xyz, False
        else:
            # a veces vienen (T,237) como (T, J*D) pero sin reshape
            if J * D == 79 * 3:
                xyz = arr.reshape(T, 79, 3).astype(np.float32)
                return xyz, True
            if J * D == 79 * 2:
                xy = arr.reshape(T, 79, 2).astype(np.float32)
                c = np.ones((T, 79, 1), dtype=np.float32)
                xyz = np.concatenate([xy, c], axis=2)
                return xyz, False
            raise ValueError(f"Forma no soportada (T={T},J={J},D={D}).")
    elif arr.ndim == 2:
        T, D = arr.shape
        if D == 237:
            xyz = arr.reshape(T, 79, 3).astype(np.float32)
            return xyz, True
        elif D == 158:
            xy = arr.reshape(T, 79, 2).astype(np.float32)
            c = np.ones((T, 79, 1), dtype=np.float32)
            xyz = np.concatenate([xy, c], axis=2)
            return xyz, False
        else:
            raise ValueError(f"Forma 2D no soportada (T={T}, D={D}).")
    else:
        raise ValueError(f"Dimensionalidad no soportada: {arr.shape}")

def _ffill_bfill_time(a: np.ndarray) -> np.ndarray:
    """Forward-fill / backward-fill temporal en NaNs por joint/coordenada."""
    # a: (T,79,2) o (T,79,3) (solo aplicamos sobre XY)
    T = a.shape[0]
    for j in range(a.shape[1]):
        for d in range(2):  # x,y
            col = a[:, j, d]
            if np.isnan(col).all():
                # todo NaN -> cero
                a[:, j, d] = 0.0
                continue
            # ffill
            idx = np.where(~np.isnan(col))[0]
            first = idx[0]; last = idx[-1]
            col[:first] = col[first]
            col[last+1:] = col[last]
            # interpolate in between
            nan_idx = np.where(np.isnan(col))[0]
            while len(nan_idx) > 0:
                # simple linear fill over single gaps
                for k in nan_idx:
                    # busca vecinos
                    l = k - 1
                    r = k + 1
                    while r < T and np.isnan(col[r]):
                        r += 1
                    if l >= 0 and r < T and not np.isnan(col[l]) and not np.isnan(col[r]):
                        col[k:r] = np.linspace(col[l], col[r], r - l + 1)[1:-1]
                nan_idx = np.where(np.isnan(col))[0]
            a[:, j, d] = col
    return a

def normalize_xy_clip(arr: np.ndarray, min_conf: float = 0.0, xy_only: bool = True) -> np.ndarray:
    """
    arr: (T,79,3) o (T,79,2) (tras ensure_xyc garantizamos (T,79,3))
    Pasos:
     1) aplica máscara de conf < min_conf en XY como NaN
     2) centra por media XY visible por frame
     3) escala por rango max(xr, yr) por frame (fallback a 1)
     4) ffill/bfill temporal para NaNs; nan_to_num; clamp
     5) aplanar a (T, 79*2) si xy_only; si no, concat c -> (T, 79*3)
    """
    xyz, _ = ensure_xyc(arr)
    T, J, _ = xyz.shape

    # 1) máscara por confianza
    c = xyz[:, :, 2]
    mask_low = c < float(min_conf)
    xy = xyz[:, :, :2].copy()
    xy[mask_low] = np.nan

    # 2-3) normalización por frame
    # evitar frames 100% NaN -> se dejarán como NaN y luego se ffill/bfill
    for t in range(T):
        xt = xy[t, :, 0]
        yt = xy[t, :, 1]
        vis = ~(np.isnan(xt) | np.isnan(yt))
        if vis.sum() < 2:
            continue
        cx = np.nanmean(xt)
        cy = np.nanmean(yt)
        xt = xt - cx
        yt = yt - cy
        xr = np.nanmax(xt) - np.nanmin(xt)
        yr = np.nanmax(yt) - np.nanmin(yt)
        scale = max(xr, yr)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        xt = xt / scale
        yt = yt / scale
        xy[t, :, 0] = xt
        xy[t, :, 1] = yt

    # 4) ffill/bfill + nan_to_num + clamp
    xyz_norm = np.concatenate([xy, c[:, :, None]], axis=2)  # (T,79,3)
    xyz_norm = _ffill_bfill_time(xyz_norm)
    xyz_norm = np.nan_to_num(xyz_norm, nan=0.0, posinf=0.0, neginf=0.0)
    xyz_norm[:, :, :2] = np.clip(xyz_norm[:, :, :2], -5.0, 5.0)

    if xy_only:
        feats = xyz_norm[:, :, :2].reshape(T, J * 2).astype(np.float32)  # (T,158)
    else:
        feats = xyz_norm.reshape(T, J * 3).astype(np.float32)           # (T,237)
    return feats

# ---------------------------
# Dataset
# ---------------------------

class SampleRow:
    def __init__(self, base: str, npy_path: str, start_f: int, end_f: int, text: str):
        self.base = base
        self.npy_path = npy_path
        self.start_f = int(max(0, start_f))
        self.end_f = int(max(self.start_f + 1, end_f))
        self.text = str(text)

class LSATDataset(Dataset):
    def __init__(self, rows: List[SampleRow], tok, xy_only=True, min_conf=0.0, fps=30.0):
        self.rows = rows
        self.tok = tok
        self.xy_only = xy_only
        self.min_conf = float(min_conf)
        self.fps = float(fps)
        self.pad_id = tok.pad_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        arr = np.load(r.npy_path)
        # robustez a cualquier formato
        feats_full = normalize_xy_clip(arr, min_conf=self.min_conf, xy_only=self.xy_only)  # (T, D)
        T = feats_full.shape[0]

        s = min(max(0, r.start_f), T - 1)
        e = min(max(s + 1, r.end_f), T)
        feats = feats_full[s:e, :]  # (t, D)
        if feats.shape[0] <= 0:
            # fallback: usar al menos 1 frame
            feats = feats_full[max(0, T // 2):max(1, T // 2 + 1), :]

        # tokens
        # mBART espera language code; forzamos id BOS en es_XX
        label_ids = self.tok(
            r.text,
            max_length=128,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        label_ids = [self.tok.lang_code_to_id.get("es_XX", self.tok.pad_token_id)] + label_ids
        y = torch.tensor(label_ids, dtype=torch.long)
        x = torch.tensor(feats, dtype=torch.float32)
        return x, y, r.base

# ---------------------------
# Collate (top-level, sin closures)
# ---------------------------

def collate_batch(batch, pad_id: int):
    """
    batch: list of (x: (t,D) tensor, y: (L,) tensor, base)
    Devuelve:
      xs    : (B, Tmax, D) float
      lens  : (B,) long
      y_pad : (B, Lmax) long (pad_id)
      key_pad: (B, Tmax) bool (True=pad)
      bases : list[str]
    """
    xs, ys, bases = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    D = xs[0].shape[1]
    Tmax = int(max(lens))
    B = len(xs)

    x_pad = torch.zeros(B, Tmax, D, dtype=torch.float32)
    key_pad = torch.ones(B, Tmax, dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x
        key_pad[i, :t] = False

    Ls = [y.shape[0] for y in ys]
    Lmax = int(max(Ls))
    y_pad = torch.full((B, Lmax), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        y_pad[i, : y.shape[0]] = y

    return x_pad, lens, y_pad, key_pad, bases

# ---------------------------
# Modelo
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        den = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: (B,T,d)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)

class KP2Text(nn.Module):
    def __init__(self, tok, d_in=158, d_model=1024, nhead=8, nlayers=8,
                 tok_name='facebook/mbart-large-50-many-to-many-mmt',
                 tgt_lang='es_XX', unfreeze_last_n_dec_layers=2, dropout=0.1):
        super().__init__()
        self.tok = tok
        self.d_in = d_in
        self.d_model = d_model

        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(d_model, dropout=dropout)

        self.decoder = MBartForConditionalGeneration.from_pretrained(tok_name)
        # idioma destino
        if hasattr(self.tok, "lang_code_to_id") and tgt_lang in self.tok.lang_code_to_id:
            bos_id = self.tok.lang_code_to_id[tgt_lang]
            self.decoder.config.forced_bos_token_id = bos_id

        # Congelar todo el decoder salvo las N últimas capas
        for p in self.decoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n_dec_layers > 0:
            try:
                dec_layers = self.decoder.model.decoder.layers
            except Exception:
                dec_layers = []
            for layer in dec_layers[-unfreeze_last_n_dec_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # layer norm final
            if hasattr(self.decoder.model.decoder, "layer_norm"):
                for p in self.decoder.model.decoder.layer_norm.parameters():
                    p.requires_grad = True
            # lm_head (proyección al vocab)
            for p in self.decoder.lm_head.parameters():
                p.requires_grad = True

    def encode(self, x, key_pad):
        # x: (B,T,d_in)
        h = self.in_proj(x)              # (B,T,d_model)
        h = self.posenc(h)               # (B,T,d_model)
        mem = self.encoder(h, src_key_padding_mask=key_pad)  # (B,T,d_model)
        return mem

    def forward(self, x, lens, y_pad, key_pad):
        """
        x     : (B,T,D)
        lens  : (B,)
        y_pad : (B,L)
        key_pad: (B,T) True=pad
        """
        mem = self.encode(x, key_pad)  # (B,T,d_model)
        out = self.decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            labels=y_pad,
        )
        return out

    @torch.no_grad()
    def generate_text(self, x, key_pad, max_len=64, beam=5):
        mem = self.encode(x, key_pad)
        gen_ids = self.decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            num_beams=beam, max_length=max_len, early_stopping=True
        )
        return gen_ids

# ---------------------------
# Optimizer / Scheduler
# ---------------------------

def build_optimizer(model: KP2Text, lr_enc=3e-4, lr_dec=1e-5, weight_decay=0.0):
    enc_params = list(model.in_proj.parameters()) + list(model.encoder.parameters()) + list(model.posenc.parameters())
    dec_params = []
    for p in model.decoder.parameters():
        if p.requires_grad and (p not in enc_params):
            dec_params.append(p)
    param_groups = [
        {'params': enc_params, 'lr': lr_enc},
        {'params': dec_params, 'lr': lr_dec},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return opt

def build_scheduler(opt, t_max: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max))

# ---------------------------
# BLEU
# ---------------------------

def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    if not SACREBLEU_OK:
        return 0.0
    bleu = BLEU(tokenize='13a')
    score = bleu.corpus_score(hyps, [refs]).score
    return float(score)

# ---------------------------
# Lectura de CSV y preparación de filas
# ---------------------------

def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, Optional[str]]:
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in col_map: return col_map[c]
        return None

    col_id = pick(['id', 'video', 'base'])
    col_text = pick(['text', 'sentence', 'caption', 'transcription', 'traduccion', 'traducción'])
    col_start = pick(['start', 'tstart', 'inicio'])
    col_end = pick(['end', 'tend', 'fin'])
    col_split = pick(['split', 'fold', 'part'])

    if col_id is None:
        raise ValueError("No se encontró columna de id/base (id|video|base).")
    if col_text is None:
        raise ValueError("No se encontró columna de texto (text|sentence|caption|...).")
    # start/end pueden faltar (se asume clip completo)
    return col_id, col_start, col_end, col_text, col_split

def prepare_rows(csv_path: str, csv_delim: str, kp_map: Dict[str, str], fps: float = 30.0) -> Tuple[List[SampleRow], List[SampleRow]]:
    df = try_read_csv(csv_path, sep=csv_delim)
    col_id, col_start, col_end, col_text, col_split = detect_columns(df)

    # normalización básica
    df[col_id] = df[col_id].astype(str)
    df[col_text] = df[col_text].astype(str)

    kept, no_npy, bad_time = 0, 0, 0
    rows_train, rows_val = [], []

    for _, row in df.iterrows():
        base = str(row[col_id]).strip()
        if base not in kp_map:
            no_npy += 1
            continue
        npy_path = kp_map[base]

        # tiempos
        if col_start is not None and col_end is not None:
            s = parse_time_any(row[col_start])
            e = parse_time_any(row[col_end])
        else:
            s, e = 0.0, None

        # leer T para recortar si hace falta
        try:
            arr = np.load(npy_path, mmap_mode='r')
            # convertir a T para clamp
            if arr.ndim == 3:
                T = arr.shape[0]
            elif arr.ndim == 2:
                T = arr.shape[0]
            else:
                T = 0
        except Exception:
            no_npy += 1
            continue

        if e is None:
            e = float(T) / fps

        if s is None or e is None:
            bad_time += 1
            continue

        # clamp a [0, T]
        s_f = int(max(0, round(s * fps)))
        e_f = int(max(0, round(e * fps)))
        if e_f <= s_f:
            e_f = min(max(s_f + 1, T), T)
            if e_f <= s_f:
                bad_time += 1
                continue

        text = row[col_text]
        sample = SampleRow(base=base, npy_path=npy_path, start_f=s_f, end_f=e_f, text=text)

        split = None
        if col_split is not None:
            split = str(row[col_split]).strip().lower()
        if split in ('val', 'valid', 'validation', 'dev'):
            rows_val.append(sample)
        elif split in ('train', '', None):
            rows_train.append(sample)
        else:
            # si split desconocido, enviar a train por defecto
            rows_train.append(sample)
        kept += 1

    plog(f"[INFO] Filtrado CSV por .npy -> kept={kept:,} | no_npy={no_npy:,} | bad_time={bad_time:,}")
    return rows_train, rows_val

# ---------------------------
# Entrenamiento / Validación
# ---------------------------

def train_one_epoch(model: KP2Text, opt, sched, ld_train, device='cuda', grad_clip=1.0, amp=True):
    model.train()
    total = 0.0
    steps = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.startswith("cuda") and torch.cuda.is_available()))
    pbar = tqdm(ld_train, desc='[Train]')
    for xs, lens, y_pad, key_pad, _ in pbar:
        xs = xs.to(device)
        y_pad = y_pad.to(device)
        key_pad = key_pad.to(device)

        with torch.amp.autocast('cuda', enabled=(amp and device.startswith("cuda") and torch.cuda.is_available())):
            out = model(xs, lens, y_pad, key_pad)
            loss = out.loss

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)

        total += float(loss.detach().cpu().item())
        steps += 1
        pbar.set_postfix({'loss': f"{total/steps:.4f}"})

    return total / max(1, steps)

@torch.no_grad()
def validate_bleu(model: KP2Text, ld_val, tok, device='cuda', beam=5, max_len=64):
    model.eval()
    refs, hyps = [], []
    for xs, lens, y_pad, key_pad, bases in tqdm(ld_val, desc='[Val]'):
        xs = xs.to(device)
        key_pad = key_pad.to(device)
        gen_ids = model.generate_text(xs, key_pad, max_len=max_len, beam=beam)
        # quitar primer token si es lang_code + limpiar pads
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # refs: decodificar labels (y_pad) sin pads
        for i in range(y_pad.size(0)):
            y = y_pad[i].tolist()
            # remove pad and leading language code if present
            y = [t for t in y if t != tok.pad_token_id]
            if len(y) > 0 and hasattr(tok, "lang_code_to_id"):
                # si el primero es un lang code, quitarlo
                lang_ids = set(tok.lang_code_to_id.values())
                if y[0] in lang_ids:
                    y = y[1:]
            ref = tok.decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            refs.append(ref.strip())
        for h in texts:
            hyps.append(h.strip())

    bleu = compute_bleu(refs, hyps) if len(refs) > 0 else 0.0
    return bleu, refs, hyps

def make_dataloaders(train_ds, val_ds, pad_id, args):
    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=getattr(args, 'pin_memory', False),
        collate_fn=lambda b: collate_batch(b, pad_id),
        drop_last=False
    )
    val_ld = DataLoader(
        val_ds, batch_size=max(1, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=getattr(args, 'pin_memory', False),
        collate_fn=lambda b: collate_batch(b, pad_id),
        drop_last=False
    )
    return train_ld, val_ld

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kp_dir', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--csv_delim', default=';')
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--pin_memory', type=lambda s: s.lower()=='true', default=False)
    ap.add_argument('--prefetch_factor', type=int, default=2)
    ap.add_argument('--persistent_workers', type=lambda s: s.lower()=='true', default=False)

    ap.add_argument('--xy_only', type=lambda s: s.lower()=='true', default=True)
    ap.add_argument('--min_conf', type=float, default=0.25)
    ap.add_argument('--fps', type=float, default=30.0)

    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--nlayers', type=int, default=8)
    ap.add_argument('--d_model', type=int, default=1024)

    ap.add_argument('--lr_encoder', type=float, default=3e-4)
    ap.add_argument('--lr_decoder', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--t_max', type=int, default=40)

    ap.add_argument('--tok_name', default='facebook/mbart-large-50-many-to-many-mmt')
    ap.add_argument('--tgt_lang', default='es_XX')
    ap.add_argument('--unfreeze_last_n_dec_layers', type=int, default=2)

    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--max_gen_len', type=int, default=64)

    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    device = args.device
    plog(f"[INFO] Dispositivo: {device}")

    kp_map = list_npy(args.kp_dir)
    plog(f"[INFO] .npy encontrados en kp_dir: {len(kp_map):,}")

    # Tokenizer
    tok = MBart50TokenizerFast.from_pretrained(args.tok_name)
    # aseguramos códigos de idioma
    if hasattr(tok, "lang_code_to_id") and args.tgt_lang in tok.lang_code_to_id:
        tok.src_lang = args.tgt_lang
        tok.tgt_lang = args.tgt_lang

    # Filtrar CSV
    train_rows, val_rows = prepare_rows(args.csv, args.csv_delim, kp_map, fps=args.fps)

    if len(train_rows) > 0 and len(val_rows) == 0:
        plog("[WARN] CSV sin partición de validación. Separa automáticamente 10% para val.")
        if len(train_rows) == 1:
            # Caso extremo: un solo ejemplo. Usar el mismo para train y val.
            val_rows = [train_rows[0]]
        else:
            n_val = max(1, int(round(0.1 * len(train_rows))))
            # asegurar al menos un ejemplo en train
            if n_val >= len(train_rows):
                n_val = len(train_rows) - 1
            idx = list(range(len(train_rows)))
            random.Random(SEED).shuffle(idx)
            val_idx = set(idx[:n_val])
            new_train = []
            new_val = []
            for i, row in enumerate(train_rows):
                (new_val if i in val_idx else new_train).append(row)
            train_rows, val_rows = new_train, new_val

    plog(f"[INFO] [train] ejemplos={len(train_rows):,} | [val] ejemplos={len(val_rows):,}")
    d_in = 158 if args.xy_only else 237
    plog(f"[INFO] d_in={d_in} | xy_only={args.xy_only} | min_conf={args.min_conf}")

    if len(train_rows) == 0 or len(val_rows) == 0:
        plog("[ERROR] Dataset vacío: train o val sin muestras.")
        return

    train_ds = LSATDataset(train_rows, tok, xy_only=args.xy_only, min_conf=args.min_conf, fps=args.fps)
    val_ds = LSATDataset(val_rows, tok, xy_only=args.xy_only, min_conf=args.min_conf, fps=args.fps)

    pad_id = tok.pad_token_id
    train_ld, val_ld = make_dataloaders(train_ds, val_ds, pad_id, args)

    # Modelo
    model = KP2Text(
        tok, d_in=d_in, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
        tok_name=args.tok_name, tgt_lang=args.tgt_lang,
        unfreeze_last_n_dec_layers=args.unfreeze_last_n_dec_layers
    ).to(device)

    opt = build_optimizer(model, lr_enc=args.lr_encoder, lr_dec=args.lr_decoder, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, t_max=args.t_max)

    best_bleu = -1.0
    best_path = osp.join(args.work_dir, 'best.pth')

    plog("\n===== Entrenamiento =====\n")
    for epoch in range(1, args.epochs + 1):
        plog(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss = train_one_epoch(model, opt, sched, train_ld, device=device, grad_clip=1.0, amp=True)
        plog(f"[Train] loss={tr_loss:.4f}")

        bleu, _, _ = validate_bleu(model, val_ld, tok, device=device, beam=args.beam, max_len=args.max_gen_len)
        plog(f"[Val] BLEU={bleu:.2f}")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), best_path)
            plog(f"[SAVE] Nuevo mejor BLEU={best_bleu:.2f} → {best_path}")

    plog(f"\n[FIN] Mejor BLEU: {best_bleu:.2f} | Checkpoint: {best_path}")

if __name__ == '__main__':
    main()
