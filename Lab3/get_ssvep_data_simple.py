from psychopy import visual, core, event
import numpy as np
import time

# === 基本參數 ===
win = visual.Window(size=(800, 800), color='black', units='pix', fullscr=False)
frequencies = [8, 10, 12, 15]  # 四個頻率 (Hz)
duration = 5  # 每次 trial 秒數  5
start_time_duration = 1  # 每次 trial 秒數
text_time_duration = 1  # 每次 trial 秒數
final_rest_time_duration = 2  # 每次 trial 秒數  2
refresh_rate = 60  # 螢幕刷新率

# === 刺激位置（左上、右上、右下、左下）===
positions = [(-300, 300), (300, 300), (300, -300), (-300, -300)]

# === 建立不同形狀的刺激 ===
stimuli = []

# 左上：方形
stimuli.append(visual.Rect(win, width=100, height=100, pos=positions[0], fillColor='red', lineColor='red'))

# 右上：圓形
stimuli.append(visual.Circle(win, radius=50, pos=positions[1], fillColor='green', lineColor='green'))

# 右下：三角形
triangle_vertices = [(0, 50), (-50, -50), (50, -50)]
stimuli.append(visual.ShapeStim(win, vertices=triangle_vertices, pos=positions[2], fillColor='blue', lineColor='blue'))

# # 左下：菱形
# diamond_vertices = [(0, 60), (60, 0), (0, -60), (-60, 0)]
# stimuli.append(visual.ShapeStim(win, vertices=diamond_vertices, pos=positions[3], fillColor='yellow', lineColor='yellow'))

# === 指示箭頭 ===
arrow = visual.TextStim(win, text='', color='white', height=50)
# arrow_dirs = ['↖', '↗', '↘', '↙']
arrow_dirs = ['↖', '↗', '↘']
labels = [0, 1, 2]

# === 開啟紀錄檔案 ===
log_file = open("5log.txt", "a")

trial_index = 0
while True:
    # np.random.seed(trial_index)
    random_num = np.random.randint(3)  # 0~2 # 不要 random 把註解的 target_idx 拿掉，然後把 random_num 註解
    print(f"random_num: {random_num}")
    target_idx = trial_index % 3  # 循環選擇刺激
    target_freq = frequencies[target_idx]

    # 空畫面休息 1 秒
    # 建立垂直與水平的矩形形成十字
    vertical_bar = visual.Rect(win, width=10, height=100, pos=[0, 0], fillColor='white', lineColor='white')
    horizontal_bar = visual.Rect(win, width=100, height=10, pos=[0, 0], fillColor='white', lineColor='white')
    # 繪製並顯示
    vertical_bar.draw()
    horizontal_bar.draw()
    win.flip()
    core.wait(start_time_duration)

    # 顯示提示箭頭
    # arrow.text = f'{arrow_dirs[target_idx]}'
    arrow.text = f'{arrow_dirs[random_num]}'
    arrow.draw()
    win.flip()
    core.wait(text_time_duration)

    # 記錄開始時間
    start_time = time.time()
    log_file.write(f"Trial {trial_index} START: {start_time:.3f} LABEL: {labels[random_num]}\n")
    log_file.flush()

    # 刺激閃爍
    clock = core.Clock()
    while clock.getTime() < duration:
        t = clock.getTime()
        for i, stim in enumerate(stimuli):
            freq = frequencies[i]
            # sin波決定閃爍
            stim.opacity = 1.0 if np.sin(2 * np.pi * freq * t) > 0 else 0.0
            stim.draw()
        win.flip()

    # 記錄結束時間
    end_time = time.time()
    log_file.write(f"Trial {trial_index} END: {end_time:.3f}\n")
    log_file.flush()

    win.flip()
    core.wait(final_rest_time_duration)

    # print(f"trial_index: {trial_index}, now dir: {arrow_dirs[target_idx]}")
    print(f"trial_index: {trial_index}, now dir: {arrow_dirs[random_num]}")
    trial_index += 1
    # 結束條件（可取消註解以便中止）
    if 'escape' in event.getKeys() or trial_index == 37:
        break

# 關閉檔案與視窗
log_file.close()
win.close()
core.quit()
