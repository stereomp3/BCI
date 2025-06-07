from psychopy import visual, core, event, monitors
import numpy as np
import time

# === 基本參數 ===
mon = monitors.Monitor(name='MyMonitor', width=52, distance=70)
mon.setSizePix([800, 800])
win = visual.Window(monitor=mon, size=(800, 800), color='black', units='pix', fullscr=False)
core.wait(1.0)
for _ in range(60):
    win.flip()
# frame_rate = win.getActualFrameRate(nIdentical=60, nMaxFrames=120, nWarmUpFrames=60)
# if frame_rate is None:
#     frame_rate = 60.0  # 預設為 60 Hz
# print(f"Actual Frame Rate: {frame_rate:.2f} Hz")

# frequencies = [7, 15, 24]  # 三個頻率 (Hz)
# frequencies = [4, 8, 15]  # 三個頻率 (Hz)
# frequencies = [5, 11, 17]  # 三個頻率 (Hz)
frequencies = [5, 17]  # 三個頻率 (Hz)
duration = 5  # 每次 trial 秒數
start_time_duration = 1
text_time_duration = 1
final_rest_time_duration = 2

# === 建立刺激圖形 ===
stimuli = []
size = 300
stimuli.append(visual.Rect(win, width=size * 2, height=size * 2, pos=(0, 0), fillColor='red', lineColor='red'))
stimuli.append(visual.Circle(win, radius=size, pos=(0, 0), fillColor='green', lineColor='green'))
triangle_vertices = [(0, size), (-size, -size), (size, -size)]
stimuli.append(visual.ShapeStim(win, vertices=triangle_vertices, pos=(0, 0), fillColor='blue', lineColor='blue'))

arrow = visual.TextStim(win, text='', color='white', height=50)
arrow_dirs = ['stay', 'up', 'down']
labels = [0, 1, 2]

log_file = open("5log.txt", "a")
trial_index = 0
max_trail = 60
stim_clock = core.Clock()
max_index = len(frequencies) # 3
while True:
    random_num = np.random.randint(max_index)
    target_idx = random_num
    # target_idx = trial_index % max_index
    target_freq = frequencies[target_idx]

    # 顯示十字固定視線
    vertical_bar = visual.Rect(win, width=10, height=100, pos=[0, 0], fillColor='white', lineColor='white')
    horizontal_bar = visual.Rect(win, width=100, height=10, pos=[0, 0], fillColor='white', lineColor='white')
    vertical_bar.draw()
    horizontal_bar.draw()
    win.flip()
    core.wait(start_time_duration)

    # 顯示提示文字
    arrow.text = f'{arrow_dirs[target_idx]}'
    arrow.draw()
    win.flip()
    core.wait(text_time_duration)

    # 記錄開始時間
    start_time = time.time()
    log_file.write(f"Trial {trial_index} START: {start_time:.3f} LABEL: {labels[target_idx]}\n")
    log_file.flush()

    # === 根據頻率產生 frame 閃爍序列 ===
    # n_frames = int(duration * frame_rate)
    # frames_per_cycle = frame_rate / target_freq
    # pattern = np.array([(i % frames_per_cycle) < (frames_per_cycle / 2) for i in range(n_frames)])

    # === 開始呈現刺激 ===
    clock = core.Clock()
    while clock.getTime() < duration:
        t = stim_clock.getTime()
        phase = (t * frequencies[target_idx]) % 1
        stimuli[target_idx].opacity = 1.0 if phase < 0.5 else 0.0
        stimuli[target_idx].draw()
        win.flip()
    # for i in range(n_frames):
    #     stimuli[target_idx].opacity = 1.0 if pattern[i] else 0.0
    #     stimuli[target_idx].draw()
    #     win.flip()

    # 結束
    end_time = time.time()
    log_file.write(f"Trial {trial_index} END: {end_time:.3f}\n")
    log_file.flush()
    win.flip()
    core.wait(final_rest_time_duration)

    print(f"trial_index: {trial_index}, now dir: {arrow_dirs[target_idx]}")
    trial_index += 1
    if 'escape' in event.getKeys() or trial_index == max_trail:
        break

log_file.close()
win.close()
core.quit()
