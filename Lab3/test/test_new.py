# -*- coding: utf-8 -*-
"""
EEG training pipeline with per‑trial Z‑score normalization.
Main changes vs. original script:
  1. After band‑pass filtering each segment, apply Z‑score per channel:
        seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
  2. Valid / train split unchanged.
  3. Ensure numpy → torch conversion uses float32.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from torch.utils.data import TensorDataset
import torch
import torch.nn.functional as F

from MI_train import EEGTrainerFineTune
from Models import ESNet, EEGNet_SSVEP
import random

# -------------------------------------------------------------

def butter_bandpass(low, high, fs, order=4):
    return butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')


# -------------------------------------------------------------

def main():
    file_path = ['data/j/01.csv', 'data/j/02.csv']
    log_path = ['data/j/01.txt', 'data/j/02.txt']
    channel_index = [2, 3, 5, 6]  # O1 / O2

    segments, labels = [], []

    for f_csv, f_log in zip(file_path, log_path):
        # ---- 1. Load raw CSV (skip meta header) -------------------------
        ts_list, eeg_list = [], []
        with open(f_csv, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(11):
                next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= max(channel_index) + 1:
                    ts_list.append(float(parts[0]))
                    eeg_list.append([float(parts[idx]) for idx in channel_index])
        eeg = np.asarray(eeg_list, dtype=np.float32)  # (N, 2)
        timestamps = np.asarray(ts_list, dtype=np.float64)

        # ---- 2. Down‑sample to 250 Hz & band‑pass -----------------------
        orig_fs, new_fs = 1000, 250
        eeg_ds = resample_poly(eeg, up=1, down=orig_fs // new_fs, axis=0)
        ts_ds = timestamps[::orig_fs // new_fs]
        b, a = butter_bandpass(1, 20, new_fs, order=4)
        eeg_f = filtfilt(b, a, eeg_ds, axis=0)

        # ---- 3. Parse log file for trial windows -----------------------
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

        # ---- 4. Convert absolute to relative seconds -------------------
        with open(f_csv, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(10):
                line = f.readline()
                if 'Record datetime' in line:
                    rec_dt_str = re.search(r'Record datetime:\s*([0-9\- :\.]+)', line).group(1).strip()
                    break
        tz = datetime.timezone(datetime.timedelta(hours=8))
        rec_epoch = datetime.datetime.strptime(rec_dt_str, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()

        # ---- 5. Slice segments, apply Z‑score ---------------------------
        for idx in sorted(trials):
            info = trials[idx]
            if {'start', 'end', 'label'} <= info.keys():
                t0 = info['start'] - rec_epoch
                t1 = info['end'] - rec_epoch
                i0 = np.searchsorted(ts_ds, t0, side='left')
                i1 = np.searchsorted(ts_ds, t1, side='right')
                seg = eeg_f[i0:i1]
                if seg.size:
                    # --- per‑trial Z‑score per channel ---
                    mu = seg.mean(axis=0, keepdims=True)
                    sig = seg.std(axis=0, keepdims=True) + 1e-8
                    seg = (seg - mu) / sig

                    segments.append(seg.astype(np.float32))
                    labels.append(info['label'])

    if not segments:
        raise RuntimeError('No valid segments collected')

    # ---- 6. Pad / trim to shortest length ------------------------------
    min_len = min(seg.shape[0] for seg in segments)
    data = np.stack([seg[:min_len] for seg in segments])  # (N, L, 2)
    data = data.transpose(0, 2, 1)  # (N, 2, L)

    # ---- 7. Train‑/val split -----------------------------------------
    valid_num = 30
    X_valid = torch.tensor(data[:valid_num], dtype=torch.float32).unsqueeze(1)  # (V, 1, 2, L)
    y_valid = F.one_hot(torch.tensor(labels[:valid_num])).float()
    X_train = torch.tensor(data[valid_num:], dtype=torch.float32).unsqueeze(1)
    y_train = F.one_hot(torch.tensor(labels[valid_num:])).float()

    ds_val = TensorDataset(X_valid, y_valid)
    ds_train = TensorDataset(X_train, y_train)

    # ---- 8. Fine‑tune ESNet ------------------------------------------
    trainer = EEGTrainerFineTune(
        model_class=EEGNet_SSVEP,  # ESNet
        ft_data=ds_train,
        ft_epochs=1000,
        batch_size=16,
        lr=1e-4,
        freeze_layers=False,
        vl_data=ds_val,
    )
    seed_n = 42

    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    trainer.init_model()
    hist = trainer.train()
    best_epoch = int(np.argmin(hist['val_loss']))
    print(f"Best epoch: {best_epoch}, val acc: {hist['val_acc'][best_epoch]:.4f}")
    print(f"Max val acc: {np.max(hist['val_acc']):.4f} @ epoch {np.argmax(hist['val_acc'])}")


if __name__ == '__main__':
    main()
