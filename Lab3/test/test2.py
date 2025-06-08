import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample

# 設定參數
fs = 1000       # 原始取樣率
fs_new = 250    # 目標取樣率
channels = ["Fp1", "Fp2", "O1", "O2"]

# Bandpass filter 函數
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)

# 讀 EEG CSV
eeg_df = pd.read_csv("test01_105944.csv")
eeg_df = eeg_df.dropna(subset=["Timestamp"])
eeg_df["Timestamp"] = eeg_df["Timestamp"].astype(float)

# 擷取 channel 資料
raw_eeg = eeg_df[channels].values  # shape: (N, 4)

# bandpass filter: 1-20 Hz
filtered = bandpass_filter(raw_eeg, 1, 20, fs)

# downsample 到 250Hz
n_samples_new = int(filtered.shape[0] * fs_new / fs)
downsampled = resample(filtered, n_samples_new)

# 可選：儲存結果
downsampled_df = pd.DataFrame(downsampled, columns=channels)
downsampled_df.to_csv("eeg_downsampled_filtered.csv", index=False)