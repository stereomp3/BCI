import time
import numpy as np
import sounddevice as sd
import csv

# === 參數設定 ===
TRIAL_DURATION = 10     # 每個 trial 持續秒數
N_TRIALS = 20          # 每種狀態的次數
LABELS = {'eyes_open': 0, 'eyes_closed': 1}
log_path = "cue_log.txt"

# === 音調播放函數 ===
def play_tone(freq, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    sd.play(tone, sr)
    sd.wait()

# === Cue 與記錄 ===
f = open(log_path, "w")
log = []

for i in range(N_TRIALS):
    for state in ['eyes_open', 'eyes_closed']:
        print(f"\nTrial {i+1} | {state.upper()}")

        # 播放提示音
        freq = 880 if state == 'eyes_open' else 440
        play_tone(freq)

        # 紀錄時間與 label
        ts = time.time()
        f.write(f"Trial {i*2+LABELS[state]} START: {time.time()} LABEL: {LABELS[state]}\n")
        # log.append((ts, LABELS[state]))

        # 等待受試者維持狀態
        time.sleep(TRIAL_DURATION)
        f.write(f"Trial {i*2+LABELS[state]} END: {time.time()}\n")

f.close()
# === 儲存 log ===
# with open(log_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["timestamp", "label"])  # 0=open, 1=closed
#     writer.writerows(log)

# print(f"\n✅ Cue log saved to {log_path}")