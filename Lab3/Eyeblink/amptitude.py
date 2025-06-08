import numpy as np
import time
import threading
from pylsl import StreamInlet, resolve_streams, resolve_byprop

# 設置參數
SAMPLE_RATE = 1000  # 采樣率，根據你設備的設定
BUFFER_SIZE = 100  # 缓衝區大小
CHANNEL_COUNT = 4  # 通道數量（這裡使用 FP1 和 FP2）

# 初始化 EEG 緩衝區
eeg_buffer = np.zeros((CHANNEL_COUNT, BUFFER_SIZE))

# 匯入上面分類邏輯
from collections import deque
from scipy.stats import pearsonr

# 參數
CORRELATION_THRESHOLD = 0.8
AMPLITUDE_THRESHOLD_LOW = 80
AMPLITUDE_THRESHOLD_HIGH = 200
AMP_DIFF_THRESHOLD = 20  # μV 內視為 "差不多"
STABILITY_SECONDS = 1.5
STABILITY_SAMPLES = int(STABILITY_SECONDS * SAMPLE_RATE)


STRONG_BLINK_THRESHOLD = 1000
MODERATE_BLINK_MIN = 300
MODERATE_BLINK_MAX = 1000
HISTORY_SIZE = 10
blink_history = deque(maxlen=HISTORY_SIZE)
# 穩定追蹤用
ssvep_counter = 0
amp_diff_history = deque(maxlen=STABILITY_SAMPLES)


def classify_stable_sync(fp1, fp2):
    global ssvep_counter, amp_diff_history

    amp1 = np.max(fp1) - np.min(fp1)
    amp2 = np.max(fp2) - np.min(fp2)
    amp_diff = abs(amp1 - amp2)
    print(f"amp_diff: {amp_diff}, amp1: {amp1}, amp2: {amp2}")

    try:
        corr, _ = pearsonr(fp1, fp2)
    except Exception:
        corr = 0.0

    amp_diff_history.append(amp_diff)

    if (
            amp1 > AMPLITUDE_THRESHOLD_HIGH and
            amp2 > AMPLITUDE_THRESHOLD_HIGH and
            corr > CORRELATION_THRESHOLD and
            np.mean(amp_diff_history) < AMP_DIFF_THRESHOLD
    ):
        ssvep_counter += 1
    else:
        ssvep_counter = 0
        amp_diff_history.clear()

    if ssvep_counter >= STABILITY_SAMPLES:
        return "SSVEP-SYNC"
    elif amp1 < AMPLITUDE_THRESHOLD_LOW and amp2 < AMPLITUDE_THRESHOLD_LOW:
        return "BASELINE"
    else:
        return "ASYMMETRIC"


# 設置 LSL inlet
def setup_lsl_inlet(stream_name="Cygnus-081015-RawEEG"):
    # Option 1: get all streams
    streams = resolve_streams()
    for stream in streams:
        print(stream.name())
    print("Resolving EEG stream...")
    streams = resolve_byprop('name', stream_name)
    inlet = StreamInlet(streams[0])
    print("Stream resolved.")
    return inlet


from collections import deque

# 參數設定
BLINK_THRESHOLD = 300  # 振幅超過此值視為 blink，可根據實測調整
AMPLITUDE_QUEUE_SIZE = 50  # queue 長度
CHECK_INTERVAL = 0.5  # 每隔多久判斷一次 blink（秒）

# 初始化資料 queue
fp1_amplitude_queue = deque(maxlen=AMPLITUDE_QUEUE_SIZE)
fp2_amplitude_queue = deque(maxlen=AMPLITUDE_QUEUE_SIZE)

last_check_time = time.time()


def read_eeg(inlet):
    global eeg_buffer, last_check_time
    last_state = None

    while True:
        sample, _ = inlet.pull_sample()
        sample_data = np.array(sample[0:2] + sample[4:6]).reshape(-1, 1)  # FP1, FP2, T7, T8
        eeg_buffer[:, :-1] = eeg_buffer[:, 1:]
        eeg_buffer[:, -1] = sample_data.flatten()

        # 計算振幅（每通道）
        amplitude_fp1 = np.max(eeg_buffer[0]) - np.min(eeg_buffer[0])
        amplitude_fp2 = np.max(eeg_buffer[1]) - np.min(eeg_buffer[1])
#        print(f"Amplitude of FP1: {amplitude_fp1:.2f} μV, FP2: {amplitude_fp2:.2f} μV")

        # 存入 queue
        fp1_amplitude_queue.append(amplitude_fp1)
        fp2_amplitude_queue.append(amplitude_fp2)

        # 每 0.5 秒判斷一次是否 blink
        current_time = time.time()

        if current_time - last_check_time >= CHECK_INTERVAL:
            last_check_time = current_time

            # 統計強/中等 blink
            fp1_strong = sum(amp > STRONG_BLINK_THRESHOLD for amp in fp1_amplitude_queue)
            fp2_strong = sum(amp > STRONG_BLINK_THRESHOLD for amp in fp2_amplitude_queue)

            fp1_moderate = sum(MODERATE_BLINK_MIN < amp < STRONG_BLINK_THRESHOLD for amp in fp1_amplitude_queue)
            fp2_moderate = sum(MODERATE_BLINK_MIN < amp < STRONG_BLINK_THRESHOLD for amp in fp2_amplitude_queue)

            half = len(fp1_amplitude_queue) // 2

            # ✅ 按優先順序判斷類別
            if fp1_strong > half and fp2_strong <= half:
                blink_class = 1  # 優先：左側強 blink
            elif fp2_strong > half and fp1_strong <= half:
                blink_class = 2  # 優先：右側強 blink
            elif fp1_moderate > half and fp2_moderate > half:
                blink_class = 3  # 中等強度同步 blink
            else:
                blink_class = 0  # 無 blink

            # 紀錄與顯示
            blink_history.append(blink_class)
            print(f"*** Blink class = {blink_class} | Hist = {list(blink_history)}")

        #time.sleep(1.0 / SAMPLE_RATE)


# 主程式
if __name__ == "__main__":
    # 設置 LSL inlet 並啟動讀取线程
    inlet = setup_lsl_inlet()
    thread1 = threading.Thread(target=read_eeg, args=(inlet,), daemon=True)
    thread1.start()

    # 等待數據流
    while True:
        time.sleep(1)
