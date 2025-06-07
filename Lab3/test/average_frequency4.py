import numpy as np

def main():
    # 1. 載入 .npz
    data = np.load('eeg_fft_dataset_ds.npz')
    fft_data = data['freq_domain']  # (n_trials, n_channels, n_freq_bins)
    labels   = data['labels']       # (n_trials,)

    n_trials, n_channels, n_freq_bins = fft_data.shape
    print(f"Loaded fft_data shape = {fft_data.shape}")
    print(f"Loaded labels   shape = {labels.shape}\n")

    # 2. 指定 O1、O2 的 channel index (請依實際通道對應填)
    o1_idx = 2
    o2_idx = 3
    if not (0 <= o1_idx < n_channels and 0 <= o2_idx < n_channels):
        raise ValueError(f"O1 或 O2 的索引 (o1_idx={o1_idx}, o2_idx={o2_idx}) 超出通道數量 (n_channels={n_channels})")

    # 3. 只算 label = 0, 1, 2
    unique_labels = [0, 1, 2]

    for lbl in unique_labels:
        # 3.1 找到屬於這個 label 的 trial indices
        idx = np.where(labels == lbl)[0]
        if idx.size == 0:
            print(f"Label {lbl} 沒有任何 trial，跳過\n" + "-"*60 + "\n")
            continue

        # 3.2 取出這些 trial 在 O1, O2 兩個 channel 的頻譜 (shape=(n_trials_lbl, n_freq_bins))
        subset = fft_data[idx, :, :]          # (n_trials_lbl, n_channels, n_freq_bins)
        o1 = subset[:, o1_idx, :]             # (n_trials_lbl, n_freq_bins)
        o2 = subset[:, o2_idx, :]             # (n_trials_lbl, n_freq_bins)

        # 4. 計算每個 trial 的「O1 與 O2 兩通道頻譜能量」
        #    這裡示範用「直接對振幅做加總」作為能量：
        #    energy_o1 = 每個 trial 的 O1 在所有頻率點振幅總和 → 形狀 (n_trials_lbl,)
        #    energy_o2 = 每個 trial 的 O2 在所有頻率點振幅總和 → 形狀 (n_trials_lbl,)
        energy_o1 = np.sum(o1, axis=1)
        energy_o2 = np.sum(o2, axis=1)

        # 4.1 如果想用「振幅平方（power）」作為更精確的頻譜能量，改為：
        # energy_o1 = np.sum(o1**2, axis=1)
        # energy_o2 = np.sum(o2**2, axis=1)

        # 4.2 接著先對 O1/O2 兩通道做平均，得到每個 trial 的「平均頻譜能量值」
        energy_mean_per_trial = (energy_o1 + energy_o2) / 2.0  # → shape = (n_trials_lbl,)

        # 5. 最後跨所有 trial 做平均，得到 label=lbl 的平均頻譜能量
        mean_energy_label = np.mean(energy_mean_per_trial)

        # 6. 印出結果
        print(f"=== Label = {lbl} ===")
        print(f"Number of trials = {idx.size}")
        print(f"O1, O2 channel indices = ({o1_idx}, {o2_idx})")
        print(f"Average spectral energy per trial (先 O1/O2 平均，再跨 trial 平均) → {mean_energy_label:.3f}\n")
        print("-"*60 + "\n")

if __name__ == "__main__":
    main()
