# -*- coding: utf-8 -*-
"""
EEG training pipeline with per-trial Z-score **and lightweight data augmentation**.
Augmentations per trial (configurable):
  • ±2-sample circular time-shift
  • additive Gaussian noise (1–3 % σ)
  • random temporal mask (~5 % length)
The original segment is kept; N_AUG augmented copies are appended.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from torch.utils.data import TensorDataset
import torch
import torch.nn.functional as F

from MI_train import EEGTrainerFineTune
from Models import ESNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHANNEL_INDEX   = [5, 6]      # O1, O2
ORIG_FS, NEW_FS = 1000, 250
LOW_HZ, HIGH_HZ = 1, 20
N_AUG           = 2           # augmented copies per trial

# ---------------------------------------------------------------------------

def butter_bandpass(low, high, fs, order=4):
    return butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')


def augment_seg(seg: np.ndarray) -> np.ndarray:
    """Return a stochastically augmented copy of `seg` (shape L×C)."""
    seg_aug = seg.copy()

    # --- 1. small circular time-shift (±2 samples) ---
    shift = np.random.randint(-2, 3)
    if shift:
        seg_aug = np.roll(seg_aug, shift, axis=0)

    # --- 2. additive Gaussian noise 1–3 % of per-channel σ ---
    noise_scale = np.random.uniform(0.01, 0.03)
    sigma = seg_aug.std(axis=0, keepdims=True) + 1e-8
    seg_aug += np.random.randn(*seg_aug.shape) * sigma * noise_scale

    # --- 3. random temporal mask (≈5 % length) ---
    L = seg_aug.shape[0]
    mask_len = max(1, int(0.05 * L))
    start = np.random.randint(0, L - mask_len)
    seg_aug[start:start + mask_len, :] = 0.0

    return seg_aug.astype(np.float32)

# ---------------------------------------------------------------------------

def main():
    file_path = ['data/j/01.csv', 'data/j/02.csv']
    log_path  = ['data/j/01.txt', 'data/j/02.txt']

    segments, labels = [], []

    for f_csv, f_log in zip(file_path, log_path):
        # ---- 1. Load raw CSV ------------------------------------------------
        ts_list, eeg_list = [], []
        with open(f_csv, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(11):
                next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= max(CHANNEL_INDEX) + 1:
                    ts_list.append(float(parts[0]))
                    eeg_list.append([float(parts[idx]) for idx in CHANNEL_INDEX])
        eeg = np.asarray(eeg_list, dtype=np.float32)          # (N, 2)
        timestamps = np.asarray(ts_list, dtype=np.float64)

        # ---- 2. Pre-process -------------------------------------------------
        eeg_ds = resample_poly(eeg, up=1, down=ORIG_FS // NEW_FS, axis=0)
        ts_ds  = timestamps[::ORIG_FS // NEW_FS]
        b, a = butter_bandpass(LOW_HZ, HIGH_HZ, NEW_FS, order=4)
        eeg_f = filtfilt(b, a, eeg_ds, axis=0)

        # ---- 3. Parse log for trials ---------------------------------------
        pattern = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?')
        trials = {}
        with open(f_log, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    idx, typ, ts = int(m.group(1)), m.group(2).lower(), float(m.group(3))
                    label = int(m.group(4)) if m.group(4) else None
                    entry = trials.setdefault(idx, {})
                    entry[typ] = ts
                    if label is not None:
                        entry['label'] = label
        if not trials:
            raise RuntimeError('No trials found')

        # ---- 4. Absolute → relative time ----------------------------------
        with open(f_csv, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(10):
                line = f.readline()
                if 'Record datetime' in line:
                    rec_dt_str = re.search(r'Record datetime:\s*([0-9\- :\.]+)', line).group(1).strip()
                    break
        tz = datetime.timezone(datetime.timedelta(hours=8))
        rec_epoch = datetime.datetime.strptime(rec_dt_str, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()

        # ---- 5. Slice, Z-score, augment ------------------------------------
        for idx in sorted(trials):
            info = trials[idx]
            if {'start', 'end', 'label'} <= info.keys():
                t0 = info['start'] - rec_epoch
                t1 = info['end']   - rec_epoch
                i0 = np.searchsorted(ts_ds, t0, side='left')
                i1 = np.searchsorted(ts_ds, t1, side='right')
                seg = eeg_f[i0:i1]
                if seg.size:
                    # Z-score per channel
                    mu  = seg.mean(axis=0, keepdims=True)
                    sig = seg.std(axis=0, keepdims=True) + 1e-8
                    seg = ((seg - mu) / sig).astype(np.float32)

                    # original + augmented copies
                    segments.append(seg)
                    labels.append(info['label'])
                    for _ in range(N_AUG):
                        segments.append(augment_seg(seg))
                        labels.append(info['label'])

    if not segments:
        raise RuntimeError('No usable segments')

    # ---- 6. Align length ---------------------------------------------------
    min_len = min(seg.shape[0] for seg in segments)
    data = np.stack([seg[:min_len] for seg in segments])      # (N, L, 2)
    data = data.transpose(0, 2, 1)                           # (N, 2, L)

    # ---- 7. Split ----------------------------------------------------------
    valid_num = 30
    X_val = torch.tensor(data[:valid_num], dtype=torch.float32).unsqueeze(1)
    y_val = F.one_hot(torch.tensor(labels[:valid_num])).float()
    X_tr  = torch.tensor(data[valid_num:], dtype=torch.float32).unsqueeze(1)
    y_tr  = F.one_hot(torch.tensor(labels[valid_num:])).float()

    ds_val = TensorDataset(X_val, y_val)
    ds_tr  = TensorDataset(X_tr, y_tr)

    # ---- 8. Fine-tune ------------------------------------------------------
    trainer = EEGTrainerFineTune(
        model_class=ESNet,
        ft_data=ds_tr,
        ft_epochs=1000,
        batch_size=16,
        lr=1e-5,
        freeze_layers=False,
        vl_data=ds_val,
    )

    trainer.init_model()
    hist = trainer.train()
    best_epoch = int(np.argmin(hist['val_loss']))
    print(f"Best epoch {best_epoch} | val acc {hist['val_acc'][best_epoch]:.4f}")
    print(f"Peak val acc {np.max(hist['val_acc']):.4f} at epoch {np.argmax(hist['val_acc'])}")


if __name__ == '__main__':
    main()