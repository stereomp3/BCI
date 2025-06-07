import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import torch.nn.functional as F
from MI_train import EEGTrainerFineTune
from Models import SCCNet, EEGNet_SSVEP
import random


def preprocess_segments_independently(segments, orig_fs=1000, new_fs=250, buffer_time=0.5):
    buffer_samples = int(buffer_time * orig_fs)
    factor = orig_fs // new_fs
    # b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
    b, a = butter(1, [1 / (0.5 * new_fs), 40 / (0.5 * new_fs)], btype='band')

    processed_segments = []
    for s in segments:
        # # --- 1. 加 buffer（鏡像 padding，讓邊界變得平滑）---
        # s_padded = np.concatenate([
        #     s[:buffer_samples][::-1],  # 前 buffer，反轉鏡像
        #     s,
        #     s[-buffer_samples:][::-1]  # 後 buffer，反轉鏡像
        # ], axis=0)

        # # --- 2. Downsample ---
        # s_ds = resample_poly(s_padded, up=1, down=factor, axis=0)

        # # --- 3. Bandpass filter ---
        # s_f = filtfilt(b, a, s_ds, axis=0)

        # # --- 4. 裁掉 downsample 後的 buffer 部分 ---
        # buffer_ds = buffer_samples // factor
        # s_final = s_f[buffer_ds : -buffer_ds]

        # --- 1. Downsample ---
        s_ds = resample_poly(s, up=1, down=factor, axis=0)

        # --- 2. 加 buffer（鏡像 padding，讓邊界變得平滑）---
        buffer_ds = buffer_samples // factor  # 記得 buffer 也要對應 downsample 後的長度
        s_padded = np.concatenate([
            s_ds[:buffer_ds][::-1],  # 前 buffer，反轉鏡像
            s_ds,
            s_ds[-buffer_ds:][::-1]  # 後 buffer，反轉鏡像
        ], axis=0)

        # --- 3. Bandpass filter ---
        s_f = filtfilt(b, a, s_padded, axis=0)

        # --- 4. 裁掉 buffer 部分 ---
        s_final = s_f[buffer_ds: -buffer_ds]
        # s_final = s_final[1000:-1000]

        # --- 5. 加入結果 ---
        if s_final.shape[0] > 0:
            processed_segments.append(s_final)
        else:
            print("Segment too short after processing; skipped.")

    return processed_segments


def safe_bandpass_filter(data, fs=250, lowcut=1.0, highcut=20.0):
    n_samples = data.shape[0]

    # 動態調整階數
    max_order = 4
    order = min(max_order, (n_samples - 1) // 3)
    if order < 1:
        raise ValueError(f"Segment too short to filter safely: {n_samples} samples")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Zero-phase filtering
    filtered = filtfilt(b, a, data, axis=0)
    return filtered


def preprocess_segments_short_safe(segments, orig_fs=1000, new_fs=250):
    factor = orig_fs // new_fs
    processed_segments = []

    for s in segments:
        # 1. Downsample
        s_ds = resample_poly(s, up=1, down=factor, axis=0)

        # 2. Safe bandpass filtering
        try:
            s_f = safe_bandpass_filter(s_ds, fs=new_fs)
            processed_segments.append(s_f)
        except ValueError as e:
            print(f"Skipping segment: {e}")

    return processed_segments


def preprocess_segments_independently_with_windows(
        segments,
        labels,
        orig_fs=1000,
        new_fs=250,
        window_sec=2.0,
        stride_sec=0.5,
        buffer_time=0.5
):
    buffer_samples = int(buffer_time * orig_fs)
    factor = orig_fs // new_fs
    win_size = int(window_sec * orig_fs)
    stride = int(stride_sec * orig_fs)
    b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')

    processed_windows = []
    processed_labels = []

    for seg, label in zip(segments, labels):
        num_samples = seg.shape[0]
        for start in range(0, num_samples - win_size + 1, stride):
            window = seg[start:start + win_size]

            if window.shape[0] <= 2 * buffer_samples:
                print("Window too short to pad; skipped.")
                continue

            # 1. 先 Downsample 原始 window
            window_ds = resample_poly(window, up=1, down=factor, axis=0)

            # # 2. 鏡像 padding（用 downsample 過的訊號來 pad）
            # buffer_ds = buffer_samples // factor
            # window_padded = np.concatenate([
            #     window_ds[:buffer_ds][::-1],
            #     window_ds,
            #     window_ds[-buffer_ds:][::-1]
            # ], axis=0)
            #
            # # 3. Bandpass filter
            # try:
            #     window_f = filtfilt(b, a, window_padded, axis=0)
            # except ValueError as e:
            #     print("Filtering failed on a window:", e)
            #     continue

            # # 1. 鏡像 padding
            # window_padded = np.concatenate([
            #     window[:buffer_samples][::-1],
            #     window,
            #     window[-buffer_samples:][::-1]
            # ], axis=0)

            # # 2. Downsample
            # window_ds = resample_poly(window_padded, up=1, down=factor, axis=0)

            # # 3. Bandpass filter
            # try:
            #     window_f = filtfilt(b, a, window_ds, axis=0)
            # except ValueError as e:
            #     print("Filtering failed on a window:", e)
            #     continue

            # 4. 去除 downsample 後的 buffer
            # buffer_ds = buffer_samples // factor
            # if window_f.shape[0] <= 2 * buffer_ds:
            #     print("Window too short after downsampling; skipped.")
            #     continue
            #
            # window_final = window_f[buffer_ds:-buffer_ds]

            # 5. 收集結果與標籤
            if window_final.shape[0] > 0:
                processed_windows.append(window_final)
                processed_labels.append(label)
            else:
                print("Window too short after final trim; skipped.")

    # for win in processed_windows:
    #     print(win.shape)

    return processed_windows, processed_labels


# def preprocess_segments_independently_with_windows(segments, labels, orig_fs=1000, new_fs=250, buffer_time=0.5, window_sec=2.0, stride_sec=1.0):
#     buffer_samples = int(buffer_time * orig_fs)
#     factor = orig_fs // new_fs
#     b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')

#     window_samples = int(window_sec * orig_fs)
#     stride_samples = int(stride_sec * orig_fs)
#     window_samples_ds = int(window_sec * new_fs)  # target length after downsample
#     processed_windows = []
#     processed_labels = []

#     for s, label in zip(segments, labels):
#         for start in range(0, s.shape[0] - window_samples + 1, stride_samples):
#             window = s[start:start + window_samples]

#             # 1. Mirror padding
#             s_padded = np.concatenate([
#                 window[:buffer_samples][::-1],
#                 window,
#                 window[-buffer_samples:][::-1]
#             ], axis=0)

#             # 2. Downsample
#             s_ds = resample_poly(s_padded, up=1, down=factor, axis=0)

#             # 3. Bandpass filter
#             s_f = filtfilt(b, a, s_ds, axis=0)

#             # 4. Remove buffer (downsampled)
#             buffer_ds = buffer_samples // factor
#             s_final = s_f[buffer_ds:-buffer_ds]

#             # 5. Check if shape is exactly expected
#             if s_final.shape[0] == window_samples_ds:
#                 processed_windows.append(s_final)
#                 processed_labels.append(label)
#             else:
#                 print(f"Skipped one window: expected {window_samples_ds}, got {s_final.shape[0]}")

#     return np.array(processed_windows), np.array(processed_labels)

def part1(file_path, log_path):
    channel_index = [2, 3, 5, 6]
    # channel_index = [5, 6]
    # channel_index = [2]
    # --- 1. 讀取 EEG 資料 (metadata跳過) ---
    timestamps_list, eeg_list = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(11): next(f)  # pass first 11 line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                timestamps_list.append(float(parts[0]))  # timestamp
                # eeg_list.append([float(x) for x in parts[2:10]])
                eeg_list.append([float(parts[x]) for x in channel_index])
    eeg = np.array(eeg_list)  # (n_samples, 8)
    timestamps = np.array(timestamps_list)  # 相對時間，秒

    # --- 1a. 解析 log.txt ---
    trials = {}
    # pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)')
    pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
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

    # --- 1b. 計算 trial 相對時間，根據 Record datetime ---
    rec_line = None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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

    # --- 1c. 切片 segments via searchsorted ---
    segments, labels = [], []
    for idx in sorted(trials):  # 37 trials
        tdict = trials[idx]
        # print(tdict)  # 1747972821.966
        if 'start' in tdict and 'end' in tdict and 'label' in tdict:
            start_rel = tdict['start'] - rec_epoch
            end_rel = tdict['end'] - rec_epoch
            # print(f"tdict['start']: {tdict['start']}, tdict['end']: {tdict['end']}")
            # 找出 start_rel 在 timesteps_ds 裡面的位置 index， side 代表從左還是右
            i0 = np.searchsorted(timestamps, start_rel, side='left')
            i1 = np.searchsorted(timestamps, end_rel, side='right')
            # print(f"i0: {i0}, i1: {i1}, start_rel: {start_rel}, end_rel: {end_rel}")
            seg = eeg[i0:i1]
            if seg.size:
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

    return segments, labels


def main():
    segments, labels = [], []

    # # file_path = 'data/wei/03.csv'
    # file_path = 'data/j/01.csv'
    # # log_path = 'data/wei/03.txt'
    # log_path = 'data/j/01.txt'
    # _segments, _labels = part1(file_path=file_path, log_path=log_path)
    # segments.extend(_segments)
    # labels.extend(_labels)

    file_path = 'data/sim/e2.csv'
    # file_path = 'data/j/03.csv'
    log_path = 'data/sim/e2.txt'
    # log_path = 'data/j/03.txt'
    _segments, _labels = part1(file_path=file_path, log_path=log_path)
    segments.extend(_segments)
    labels.extend(_labels)

    # # file_path = 'data/wei/02.csv'
    # file_path = 'data/j/02.csv'
    # # log_path = 'data/wei/02.txt'
    # log_path = 'data/j/02.txt'
    # _segments, _labels = part1(file_path=file_path, log_path=log_path)
    # segments.extend(_segments)
    # labels.extend(_labels)


    # file_path = 'data/wei/03.csv'
    # log_path = 'data/wei/03.txt'
    # _segments, _labels = part1(file_path=file_path, log_path=log_path)
    # segments.extend(_segments)
    # labels.extend(_labels)

    # 統一長度為最短 segment
    min_len = min(s.shape[0] for s in segments)
    segments = [s[:min_len] for s in segments]

    # --- 2. Downsample & 濾波 ---
    # segments_f = preprocess_segments_independently(segments=segments, orig_fs=1000, new_fs=1000, buffer_time=0.5)
    # labels_f = labels
    # segments_f = preprocess_segments_short_safe(segments=segments, orig_fs=1000, new_fs=250)
    segments_f, labels_f = preprocess_segments_independently_with_windows(segments=segments, labels=labels, orig_fs=1000, new_fs=1000, window_sec=1.5, stride_sec=0.5, buffer_time=0.5)
    # orig_fs, new_fs = 1000, 250
    # factor = orig_fs // new_fs
    # for s in segments:
    #     s_ds = resample_poly(s, up=1, down=factor, axis=0)
    #     # timestamps_ds = timestamps[::factor]  # 83250 total timestamps
    #     b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
    #     s_f = filtfilt(b, a, s_ds, axis=0)  # # len(eeg_f) 83250 total sample
    #     segments_f.append(s_f)

    segments = np.stack(segments)
    segments_f = np.stack(segments_f)  # (num_trials, time, channels)
    # labels = np.array(labels)
    labels_f = np.array(labels_f)

    print(segments_f.shape, labels_f.shape)

    # X = torch.tensor(np.transpose(np.stack([s[:min_len] for s in segments]), (0, 2, 1)))
    # (n_trials, min_len, 4) -> (37, 4, 1249)
    # valid_num = 7  # 7/37
    # valid_num = int(len(segments_f) * 30 / 180)  # 7/37 # best: 19 (0.684)
    valid_num = 100  # 7/37 # best: 19 (0.684)
    # valid_num = int(len(segments_f) * 30 / 180)  # 7/37 # best: 19 (0.684)
    # data_x = np.transpose(np.stack([s[:min_len] for s in segments]), (0, 2, 1))
    data_x = np.transpose(segments_f, (0, 2, 1))

    X_valid = torch.tensor(data_x)[len(data_x) - valid_num::].unsqueeze(1)  # torch.Size([37-valid_num, 1, 4, 1249])
    # y_valid = F.one_hot(torch.from_numpy(np.array(labels)[len(data_x) - valid_num:].squeeze()).long())  # torch.Size([30, 3])
    y_valid = torch.from_numpy(np.array(labels_f)[len(data_x) - valid_num:]).long()
    print(f"X_valid.shape: {X_valid.shape}, y_valid.shape: {y_valid.shape}")
    dataset_valid = TensorDataset(X_valid, y_valid)

    X = torch.tensor(data_x)[:len(data_x) - valid_num:].unsqueeze(1)
    # X = torch.tensor(data_x).unsqueeze(1)
    # y = F.one_hot(torch.from_numpy(np.array(labels)[:valid_num].squeeze()).long())
    y = torch.from_numpy(np.array(labels_f)[:len(data_x) - valid_num]).long()
    # y = torch.from_numpy(np.array(labels_f)).long()
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")  # [torch.Size([1, 4, 1249]), torch.Size([3])]
    dataset = TensorDataset(X, y)

    print([dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape])

    trainer = EEGTrainerFineTune(
        model_class=EEGNet_SSVEP,
        ft_data=dataset,
        ft_epochs=500,
        batch_size=16,
        lr=0.0001,
        freeze_layers=False,
        vl_data=dataset_valid
    )

    seed_n = 800  # 405

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


if __name__ == '__main__':
    main()
