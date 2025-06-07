import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import datetime, re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


def main():
    file_path = 'test_backup.csv'
    log_path = 'log.txt'
    channel_index = [2, 3, 5, 6]
    # --- 1. 讀取 EEG 資料 (metadata跳過) ---
    timestamps_list, eeg_list = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(11): next(f)  # pass first 11 line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                timestamps_list.append(float(parts[0]))
                # eeg_list.append([float(x) for x in parts[2:10]])
                eeg_list.append([float(parts[x]) for x in channel_index])
    eeg = np.array(eeg_list)  # (n_samples, 8)
    timestamps = np.array(timestamps_list)  # 相對時間，秒

    # --- 2. Downsample & 濾波 ---
    orig_fs, new_fs = 1000, 250
    factor = orig_fs // new_fs
    eeg_ds = resample_poly(eeg, up=1, down=factor, axis=0)
    timestamps_ds = timestamps[::factor]  # 83250 total timestamps
    b, a = butter(4, [1 / (0.5 * new_fs), 20 / (0.5 * new_fs)], btype='band')
    eeg_f = filtfilt(b, a, eeg_ds, axis=0)  # # len(eeg_f) 83250 total sample

    # --- 3. 解析 log.txt ---
    trials = {}
    pat = re.compile(r'Trial\s+(\d+)\s+(START|END):\s*([\d\.]+)')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                idx = int(m.group(1))
                typ = m.group(2).lower()
                ts = float(m.group(3))
                trials.setdefault(idx, {})[typ] = ts
                # print(f"line: {line}, idx: {idx}, typ: {typ}, ts: {ts}")
    if not trials:
        raise ValueError('未解析到 trial 資訊')
    # print(trials)
    # --- 4. 計算 trial 相對時間，根據 Record datetime ---
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

    # --- 5. 切片 segments via searchsorted ---
    segments, labels = [], []
    for idx in sorted(trials):  # 37 trials
        tdict = trials[idx]
        # print(tdict) 1747972821.966
        if 'start' in tdict and 'end' in tdict:
            start_rel = tdict['start'] - rec_epoch
            end_rel = tdict['end'] - rec_epoch
            # print(f"tdict['start']: {tdict['start']}, tdict['end']: {tdict['end']}")
            # 找出 start_rel 在 timesteps_ds 裡面的位置 index， side 代表從左還是右
            i0 = np.searchsorted(timestamps_ds, start_rel, side='left')
            i1 = np.searchsorted(timestamps_ds, end_rel, side='right')
            print(f"i0: {i0}, i1: {i1}, start_rel: {start_rel}, end_rel: {end_rel}")
            seg = eeg_f[i0:i1]
            if seg.size:
                segments.append(seg)
                labels.append(idx % 3)
    if not segments:
        raise ValueError('No valid segments found')

    # 統一長度為最短 segment
    min_len = min(s.shape[0] for s in segments)
    X = np.stack([s[:min_len] for s in segments])  # (n_trials, min_len, 8)
    y = np.array(labels)

    # --- 6. Flatten & 分類 Pipeline ---
    X_flat = X.reshape(X.shape[0], -1)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_flat, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print('5-fold accuracies:', scores)
    print('Mean accuracy: {:.2%}'.format(scores.mean()))


if __name__ == '__main__':
    main()
