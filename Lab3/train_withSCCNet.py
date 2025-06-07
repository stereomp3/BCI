import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import torch.nn.functional as F
from MI_train import EEGTrainerFineTune
from Models import SCCNet, ESNet, EEGNet_SSVEP
import random

def main():
    file_path = ['data/j/01.csv', 'data/j/02.csv']
    log_path = ['data/j/01.txt', 'data/j/02.txt']
    channel_index = [2, 3, 5, 6]
    # channel_index = [5, 6]
    segments, labels = [], []
    for i in range(len(file_path)):
        # --- 1. 讀取 EEG 資料 (metadata跳過) ---
        timestamps_list, eeg_list = [], []
        with open(file_path[i], 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(11): next(f)  # pass first 11 line
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    timestamps_list.append(float(parts[0]))  # timestamp
                    # eeg_list.append([float(x) for x in parts[2:10]])
                    eeg_list.append([float(parts[x]) for x in channel_index])
        eeg = np.array(eeg_list)  # (n_samples, 8)
        timestamps = np.array(timestamps_list)  # 相對時間，秒

        # --- 2. Downsample & 濾波 --- # 可以把這邊移到切完後再切完濾波
        orig_fs, new_fs = 1000, 250
        factor = orig_fs // new_fs
        # b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
        # eeg = filtfilt(b, a, eeg, axis=0)  # # len(eeg_f) 83250 total sample
        eeg_ds = resample_poly(eeg, up=1, down=factor, axis=0)
        timestamps_ds = timestamps[::factor]  # 83250 total timestamps
        eeg_f = eeg_ds

        # b, a = butter(4, [1 / (0.5 * 1000), 20 / (0.5 * 1000)], btype='band')
        # eeg_f = filtfilt(b, a, eeg, axis=0)  # # len(eeg_f) 83250 total sample

        # b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
        # eeg_f = filtfilt(b, a, eeg_ds, axis=0)  # # len(eeg_f) 83250 total sample
        # --- 3. 解析 log.txt ---
        trials = {}
        # pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)')
        pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?')
        with open(log_path[i], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.match(line.strip())
                if m:
                    idx = int(m.group(1))
                    typ = m.group(2).lower()
                    ts = float(m.group(3))
                    label = int(m.group(4)) if m.group(4) is not None else None

                    trial_entry = trials.setdefault(idx, {})
                    trial_entry[typ] = ts
                    if label is not None:
                        trial_entry['label'] = label
                    # print(f"line: {line.strip()}, idx: {idx}, typ: {typ}, ts: {ts}, label: {label}")
        if not trials:
            raise ValueError('未解析到 trial 資訊')
        # --- 4. 計算 trial 相對時間，根據 Record datetime ---
        rec_line = None
        with open(file_path[i], 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(10):
                line = f.readline()
                if 'Record datetime' in line:
                    rec_line = line
                    break
        if rec_line is None:
            raise ValueError('找不到 Record datetime')
        m_dt = re.search(r'Record datetime:\s*([0-9\- :\.]+)', rec_line)
        tz = datetime.timezone(datetime.timedelta(hours=8))
        rec_dt = datetime.datetime.strptime(m_dt.group(1).strip(), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz)
        rec_epoch = rec_dt.timestamp()
        print(m_dt.group(1).strip())
        # print(trials)
        # nor_num = 1747972819.97

        # --- 5. 切片 segments via searchsorted ---

        for idx in sorted(trials):  # 37 trials
            tdict = trials[idx]
            # print(tdict)  # 1747972821.966
            if 'start' in tdict and 'end' in tdict and 'label' in tdict:
                start_rel = tdict['start'] - rec_epoch
                end_rel = tdict['end'] - rec_epoch
                # print(f"tdict['start']: {tdict['start']}, tdict['end']: {tdict['end']}")
                # 找出 start_rel 在 timesteps_ds 裡面的位置 index， side 代表從左還是右
                i0 = np.searchsorted(timestamps_ds, start_rel, side='left')
                i1 = np.searchsorted(timestamps_ds, end_rel, side='right') + 1  # 1
                # print(f"i0: {i0}, i1: {i1}, start_rel: {start_rel}, end_rel: {end_rel}")
                seg = eeg_f[i0:i1]
                if seg.size:
                    b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
                    seg = filtfilt(b, a, seg, axis=0)  # # len(eeg_f) 83250 total sample
                    segments.append(seg)
                    labels.append(tdict['label'])
            # if 'start' in tdict and 'end' in tdict: # use no label
            #     start_rel = tdict['start'] - rec_epoch
            #     end_rel = tdict['end'] - rec_epoch
            #     i0 = np.searchsorted(timestamps_ds, start_rel, side='left')
            #     i1 = np.searchsorted(timestamps_ds, end_rel, side='right')
            #     # print(f"i0: {i0}, i1: {i1}, start_rel: {start_rel}, end_rel: {end_rel}")
            #     seg = eeg_f[i0:i1]
            #     if seg.size:
            #         segments.append(seg)
            #         labels.append(idx % 3)
        if not segments:
            raise ValueError('No valid segments found')

    # 統一長度為最短 segment
    min_len = min(s.shape[0] for s in segments)
    print(min_len)
    # X = torch.tensor(np.transpose(np.stack([s[:min_len] for s in segments]), (0, 2, 1)))
    # (n_trials, min_len, 4) -> (37, 4, 1249)
    valid_num = 60  # 7/37
    data_x = np.transpose(np.stack([s[:min_len] for s in segments]), (0, 2, 1))

    X_valid = torch.tensor(data_x)[:valid_num:].unsqueeze(1)  # torch.Size([37-valid_num, 1, 4, 1249]) 前面 7 個
    y_valid = F.one_hot(torch.from_numpy(np.array(labels)[:valid_num].squeeze()).long())  # torch.Size([30, 3])
    print(f"X_valid.shape: {X_valid.shape}, y_valid.shape: {y_valid.shape}")
    dataset_valid = TensorDataset(X_valid, y_valid)

    X = torch.tensor(data_x)[valid_num::].unsqueeze(1)  # 後面 30
    # X = torch.tensor(data_x).unsqueeze(1)
    y = F.one_hot(torch.from_numpy(np.array(labels)[valid_num:].squeeze()).long())
    # y = F.one_hot(torch.from_numpy(np.array(labels).squeeze()).long())
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")  # [torch.Size([1, 4, 1249]), torch.Size([3])]
    dataset = TensorDataset(X, y)

    print([dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape])

    trainer = EEGTrainerFineTune(
        model_class=EEGNet_SSVEP,  # ESNet SCCNet EEGNet_SSVEP
        ft_data=dataset,
        ft_epochs=1000,
        batch_size=16,
        lr=1e-4,
        freeze_layers=False,
        vl_data=dataset_valid
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
    best_epoch = np.argmin(hist["val_loss"])
    print(f"best_epoch_ft: {best_epoch}, val acc: {hist['val_acc'][best_epoch]}")
    print(f"best_acc: {best_epoch}, val acc: {hist['val_acc'][np.argmax(hist['val_acc'])]}")


if __name__ == '__main__':
    main()
