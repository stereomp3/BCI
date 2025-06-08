import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re

def main():
    file_path = '../data/wei/04.csv'
    log_path  = '../data/wei/04.txt'
    channel_index = [2, 3, 5, 6]  # 原始四個通道在 CSV 的欄位索引
    orig_fs, new_fs = 1000, 250   # 原始取樣率 1000Hz，下採樣到 250Hz
    
    # --- 1. 讀 CSV（跳過前 11 行 metadata），取出 timestamp + 4 個 channel 資料 ---
    timestamps_list, eeg_list = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(11):
            next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= max(channel_index) + 1:
                timestamps_list.append(float(parts[0])) 
                eeg_list.append([float(parts[x]) for x in channel_index])
    eeg         = np.array(eeg_list, dtype=np.float32)          # shape = (total_samples, 4)
    timestamps  = np.array(timestamps_list, dtype=np.float64)   # shape = (total_samples,)
    
    # --- 2. Downsample & Bandpass filter（先 down 到 250Hz，再濾 1–20 Hz）---
    factor   = orig_fs // new_fs  # = 1000/250 = 4
    eeg_ds   = resample_poly(eeg, up=1, down=factor, axis=0)    # downsample 後 shape = (total_samples/4, 4)
    timestamps_ds = timestamps[::factor]                        # 下採樣後的時間戳 array
    
    # 4th-order bandpass: 1.–20Hz
    b, a = butter(4, [1/(0.5*new_fs), 20/(0.5*new_fs)], btype='band')
    eeg_f = filtfilt(b, a, eeg_ds, axis=0)                      # shape = (n_samples_ds, 4)
    
    # --- 3. 解析 log.txt，擷取每個 trial 的 start, end, label ---
    trials = {}
    pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            idx   = int(m.group(1))
            typ   = m.group(2).lower()           # 'start' or 'end'
            ts    = float(m.group(3))
            label = int(m.group(4)) if m.group(4) is not None else None
            
            entry = trials.setdefault(idx, {})
            entry[typ] = ts
            if label is not None:
                entry['label'] = label
    
    if not trials:
        raise ValueError('未解析到任何 trial 資訊 (log 檔內沒有 START/END)')
    
    # --- 4. 從 CSV 的 Record datetime 拿到一個 epoch 基準，用來計算相對時間 ---
    rec_line = None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(10):
            line = f.readline()
            if 'Record datetime' in line:
                rec_line = line
                break
    if rec_line is None:
        raise ValueError('找不到 "Record datetime" 行，無法對齊時間')
    
    # 範例文字: "Record datetime: 2025-06-03 12:00:00.123"
    m_dt = re.search(r'Record datetime:\s*([0-9\- :\.]+)', rec_line)
    if not m_dt:
        raise ValueError('Record datetime 無法解析')
    tz = datetime.timezone(datetime.timedelta(hours=8))
    rec_dt    = datetime.datetime.strptime(m_dt.group(1).strip(), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz)
    rec_epoch = rec_dt.timestamp()  # 這是群起始時間的 epoch（秒）
    
    # --- 5. 用 searchsorted 在下採樣後的 timestamps_ds 上，找每個 trial 的區段 → segments + labels ---
    segments, labels = [], []
    for idx in sorted(trials.keys()):  # 應該有 37 個 key
        tdict = trials[idx]
        if 'start' in tdict and 'end' in tdict and 'label' in tdict:
            start_rel = tdict['start'] - rec_epoch
            end_rel   = tdict['end']   - rec_epoch
            i0 = np.searchsorted(timestamps_ds, start_rel, side='left')
            i1 = np.searchsorted(timestamps_ds, end_rel,   side='right')
            
            # 如果切到了範圍內，就把這一段 eeg_f[i0:i1] 當成一個 trial
            seg = eeg_f[i0:i1, :]  # shape = (n_samples_trial_i, 4)
            if seg.shape[0] > 0:
                segments.append(seg)
                labels.append(tdict['label'])
            else:
                # 如果這 trial 前後沒撈到資料，就略過（正常情況下不會發生）
                print(f"Warning: 第 {idx} 號 trial 在下採樣後找不到資料範圍 (i0={i0}, i1={i1})，已跳過。")
    
    if len(segments) == 0:
        raise ValueError('經過切片後，Segments 為空！請檢查 log.txt vs CSV 的對應是否正常。')
    print(f"成功切出 {len(segments)} 個 trial (labels: {np.unique(labels)})\n")
    
    # --- 6. 先找出所有 trial 中最短的長度，將它們裁成一樣長度，方便做批次 FFT ---
    n_trials = len(segments)  # 預期應該為 37
    min_len  = min(seg.shape[0] for seg in segments)
    print(f"所有 trial 裡最短的 sample 數 = {min_len}，下面會把每個 trial 裁到這個長度。\n")
    
    # 把 segments list 變成一個 numpy array: (n_trials, min_len, n_channels)
    # 並裁切每個 trial 到前 min_len 個 sample
    data_stack = np.stack([seg[:min_len, :] for seg in segments], axis=0)
    # 現在 data_stack.shape = (n_trials, min_len, 4)
    
    # 把維度改成 (n_trials, n_channels, n_samples)，符合一般做 FFT 的輸入格式
    data_x = np.transpose(data_stack, (0, 2, 1))
    # 現在 data_x.shape = (n_trials, 4, min_len)
    n_trials, n_channels, n_samples = data_x.shape
    print(f"[時域資料] data_x shape = {data_x.shape}  # (n_trials, n_channels, n_samples)\n")
    
    # --- 7. 對每個 trial、每個 channel 做 rFFT，取振幅 (magnitude) ---
    # 計算頻率軸：min_len 個點，每點間隔是 1/250 秒
    freqs = np.fft.rfftfreq(n_samples, d=1.0/new_fs)    # shape = (n_freq_bins,)
    n_freq_bins = freqs.shape[0]
    
    # 建一個空陣列放頻域資料
    fft_data = np.zeros((n_trials, n_channels, n_freq_bins), dtype=np.float32)
    for t in range(n_trials):
        for ch in range(n_channels):
            spec = np.fft.rfft(data_x[t, ch, :])   # 回傳複數型 array, 長度 = n_freq_bins
            fft_data[t, ch, :] = np.abs(spec)      # 取振幅
    
    print(f"[頻域資料] fft_data shape = {fft_data.shape}  # (n_trials, n_channels, n_freq_bins)\n")
    
    # --- 8. 把 labels 轉成 numpy array → shape = (n_trials,) ---
    labels_arr = np.array(labels, dtype=np.int32)
    print(f"[標籤] labels_arr shape = {labels_arr.shape}, unique = {np.unique(labels_arr)}\n")
    
    # --- 9. 最後把 time_domain, freq_domain, freqs, labels 一起存成 .npz 檔 ---
    np.savez(
        'eeg_fft_dataset_ds.npz',
        time_domain = data_x,   # (n_trials, n_channels, n_samples)
        freq_domain = fft_data, # (n_trials, n_channels, n_freq_bins)
        freqs       = freqs,    # (n_freq_bins,)
        labels      = labels_arr# (n_trials,)
    )
    print("已儲存：eeg_fft_dataset_ds.npz")

if __name__ == '__main__':
    main()
