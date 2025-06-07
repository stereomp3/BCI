import random
import time
from psychopy import monitors, visual, core, event

# def multi_flicker(frequencies, duration_sec=10, window_size=(800, 800)):
#     mon = monitors.Monitor(name='MyMonitor', width=52, distance=70)  # 實際螢幕寬度(公分)、觀看距離(公分)
#     mon.setSizePix([1920, 1080])  # 解析度
#     # mon.save()
    
#     win = visual.Window(monitor=mon, size=window_size, color='grey', units='pix', fullscr=False)
    
#     # 等待螢幕初始化穩定
#     core.wait(1.0)

#     # 預熱畫面，避免剛開始掉幀
#     for _ in range(60):  # 約一秒（假設60Hz，實際會根據刷新率跑 1 秒）
#         win.flip()
        
#     # 顯示中央十字 3 秒
#     fixation = visual.TextStim(win, text='+', color='white', height=200)
#     fixation_clock = core.Clock()
#     while fixation_clock.getTime() < 3.0:
#         fixation.draw()
#         win.flip()
        
#     refresh_rate = win.getActualFrameRate()
#     if refresh_rate is None:
#         refresh_rate = 60  # 預設 60Hz
#     print(f"Refresh rate: {refresh_rate:.2f} Hz")

#     # 四個象限中心點位置
#     half_w, half_h = window_size[0]//2, window_size[1]//2
#     positions = [
#         (-half_w//2,  half_h//2),  # 左上
#         ( half_w//2,  half_h//2),  # 右上
#         (-half_w//2, -half_h//2),  # 左下
#         ( half_w//2, -half_h//2),  # 右下
#     ]

#     # 建立四個正方形刺激物 (大小100x100 px)
#     size = 100
#     stimuli = []
#     for pos in positions:
#         stim = visual.Rect(win, width=size, height=size, fillColor='black', lineColor='black', pos=pos)
#         stimuli.append(stim)

#     # 對前三個閃爍頻率計算週期幀數，第四個不閃爍（持續白色）
#     frames_per_cycle = []
#     for f in frequencies:
#         if f == 0:
#             frames_per_cycle.append(None)
#         else:
#             frames_per_cycle.append(int(round(refresh_rate / f)))

#     frame_counter = 0
#     stim_clock = core.Clock()

#     while stim_clock.getTime() < duration_sec:
#         for i, stim in enumerate(stimuli):
#             freq = frequencies[i]
#             if freq == 0:
#                 # 不閃爍，持續白色
#                 stim.fillColor = 'white'
#             else:
#                 cycle = frames_per_cycle[i]
#                 half_cycle = cycle // 2
#                 # 判斷該幀是否 ON (白色) 或 OFF (黑色)
#                 if frame_counter % cycle < half_cycle:
#                     stim.fillColor = 'white'
#                 else:
#                     stim.fillColor = 'black'
#             stim.draw()

#         win.flip()
#         frame_counter += 1

#         if event.getKeys(keyList=["escape"]):
#             break
    
#     # 黑畫面 4 秒
#     black_bg = visual.Rect(win, width=window_size[0], height=window_size[1], fillColor='black', lineColor='black')
#     rest_clock = core.Clock()
#     while rest_clock.getTime() < 4.0:
#         black_bg.draw()
#         win.flip()

#     win.close()
#     core.wait(1.0)
#     core.quit()
    
# def multi_flicker(frequencies, duration_sec=10, window_size=(800, 800), n_repeats=3, labels=[], file=None):
#     mon = monitors.Monitor(name='MyMonitor', width=52, distance=70)  # 實際螢幕寬度(公分)、觀看距離(公分)
#     mon.setSizePix([1920, 1080])  # 解析度
    
#     win = visual.Window(monitor=mon, size=window_size, color='grey', units='pix', fullscr=False)
    
#     # 等待螢幕初始化穩定
#     core.wait(1.0)

#     # 預熱畫面，避免剛開始掉幀
#     for _ in range(60):
#         win.flip()

#     # 四個象限中心點位置
#     half_w, half_h = window_size[0]//2, window_size[1]//2
#     positions = [
#         (-half_w//2,  half_h//2),  # 左上
#         ( half_w//2,  half_h//2),  # 右上
#         (-half_w//2, -half_h//2),  # 左下
#         ( half_w//2, -half_h//2),  # 右下
#     ]

#     # 建立四個正方形刺激物 (大小100x100 px)
#     size = 100
#     stimuli = []
#     for pos in positions:
#         stim = visual.Rect(win, width=size, height=size, fillColor='black', lineColor='black', pos=pos)
#         stimuli.append(stim)

#     refresh_rate = win.getActualFrameRate()
#     if refresh_rate is None:
#         refresh_rate = 60  # 預設 60Hz
#     print(f"Refresh rate: {refresh_rate:.2f} Hz")

#     frames_per_cycle = []
#     for f in frequencies:
#         if f == 0:
#             frames_per_cycle.append(None)
#         else:
#             frames_per_cycle.append(int(round(refresh_rate / f)))

#     fixation = visual.TextStim(win, text='+', color='white', height=200)
#     top_left = visual.TextStim(win, text='↖', color='white', height=200)
#     top_right = visual.TextStim(win, text='↗', color='white', height=200)
#     bottom_left = visual.TextStim(win, text='↙', color='white', height=200)
#     bottom_right = visual.TextStim(win, text='↘', color='white', height=200)
#     arrow_stims = [top_left, top_right, bottom_left, bottom_right]
#     black_bg = visual.Rect(win, width=window_size[0], height=window_size[1], fillColor='black', lineColor='black')

#     for repeat in range(n_repeats):
#         print(f"Trial {repeat} FIXATION START: {time.time()}")
#         # Fixation 十字 3秒
#         fixation_clock = core.Clock()
#         while fixation_clock.getTime() < 3.0:
#             fixation.draw()
#             win.flip()
#             if event.getKeys(keyList=["escape"]):
#                 win.close()
#                 core.wait(1.0)
#                 core.quit()
#                 return
            
#         print(f"Trial {repeat} CUE START: {time.time()} LABEL: {labels[repeat]}")
#         # Cue 3 秒
#         cue_clock = core.Clock()
#         while cue_clock.getTime() < 3.0:
#             arrow_stims[labels[repeat]].draw()
#             win.flip()
#             if event.getKeys(keyList=["escape"]):
#                 win.close()
#                 core.wait(1.0)
#                 core.quit()
#                 return
        
#         print(f"Trial {repeat} STIMULI START: {time.time()} LABEL: {labels[repeat]}")
#         # 閃爍刺激 duration_sec 秒
#         frame_counter = 0
#         stim_clock = core.Clock()
#         file.write(f"Trial {repeat} START: {time.time()} LABEL: {labels[repeat]}\n")
#         while stim_clock.getTime() < duration_sec:
#             for i, stim in enumerate(stimuli):
#                 freq = frequencies[i]
#                 if freq == 0:
#                     stim.fillColor = 'white'
#                 else:
#                     cycle = frames_per_cycle[i]
#                     half_cycle = cycle // 2
#                     if frame_counter % cycle < half_cycle:
#                         stim.fillColor = 'white'
#                     else:
#                         stim.fillColor = 'black'
#                 stim.draw()

#             win.flip()
#             frame_counter += 1

#             if event.getKeys(keyList=["escape"]):
#                 win.close()
#                 core.wait(1.0)
#                 core.quit()
#                 return
#         file.write(f"Trial {repeat} END: {time.time()}\n")

#         print(f"Trial {repeat} REST START: {time.time()}")
#         # 黑畫面休息 4秒
#         rest_clock = core.Clock()
#         while rest_clock.getTime() < 4.0:
#             black_bg.draw()
#             win.flip()
#             if event.getKeys(keyList=["escape"]):
#                 win.close()
#                 core.wait(1.0)
#                 core.quit()
#                 return
            
#         print(f"Trial {repeat} END: {time.time()}")
        
def multi_flicker(frequencies, duration_sec=10, window_size=(800, 800), n_repeats=3, labels=[], file=None):
    mon = monitors.Monitor(name='MyMonitor', width=52, distance=70)
    mon.setSizePix([1920, 1080])
    win = visual.Window(monitor=mon, size=window_size, color='grey', units='pix', fullscr=False)
    
    core.wait(1.0)
    for _ in range(60):
        win.flip()

    half_w, half_h = window_size[0]//2, window_size[1]//2
    positions = [
        (-half_w//2,  half_h//2),
        ( half_w//2,  half_h//2),
        (-half_w//2, -half_h//2),
        ( half_w//2, -half_h//2),
    ]

    size = 100
    stimuli = []
    for pos in positions:
        stim = visual.Rect(win, width=size, height=size, fillColor='black', lineColor='black', pos=pos)
        stimuli.append(stim)

    fixation = visual.TextStim(win, text='+', color='white', height=200)
    arrow_stims = [
        visual.TextStim(win, text='↖', color='white', height=200),
        visual.TextStim(win, text='↗', color='white', height=200),
        visual.TextStim(win, text='↙', color='white', height=200),
        visual.TextStim(win, text='↘', color='white', height=200),
    ]
    black_bg = visual.Rect(win, width=window_size[0], height=window_size[1], fillColor='black', lineColor='black')

    for repeat in range(n_repeats):
        print(f"Trial {repeat} FIXATION START: {time.time()}")
        fixation_clock = core.Clock()
        while fixation_clock.getTime() < 3.0:
            fixation.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                win.close(); core.wait(1.0); core.quit(); return
            
        print(f"Trial {repeat} CUE START: {time.time()} LABEL: {labels[repeat]}")
        cue_clock = core.Clock()
        while cue_clock.getTime() < 3.0:
            arrow_stims[labels[repeat]].draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                win.close(); core.wait(1.0); core.quit(); return
        
        print(f"Trial {repeat} STIMULI START: {time.time()} LABEL: {labels[repeat]}")
        stim_clock = core.Clock()
        file.write(f"Trial {repeat} START: {time.time()} LABEL: {labels[repeat]}\n")

        while stim_clock.getTime() < duration_sec:
            t = stim_clock.getTime()
            for i, stim in enumerate(stimuli):
                f = frequencies[i]
                if f == 0:
                    stim.fillColor = 'white'
                else:
                    phase = (t * f) % 1
                    stim.fillColor = 'white' if phase < 0.5 else 'black'
                stim.draw()
            win.flip()

            if event.getKeys(keyList=["escape"]):
                win.close(); core.wait(1.0); core.quit(); return

        file.write(f"Trial {repeat} END: {time.time()}\n")

        print(f"Trial {repeat} REST START: {time.time()}")
        rest_clock = core.Clock()
        while rest_clock.getTime() < 4.0:
            black_bg.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                win.close(); core.wait(1.0); core.quit(); return
        print(f"Trial {repeat} END: {time.time()}")

if __name__ == "__main__":
    # 頻率列表，最後一個是 0 表示不閃爍
    freqs = [8, 13, 15, 0]
    freqs = [2, 2, 2, 2]
    duration_sec = 100
    # window_size = (1536, 864)
    window_size = (800, 800)
    n_times = 12
    n_repeats = n_times * 4
    labels = [i % 4 for i in range(n_repeats)]
    random.shuffle(labels)
    file = open('log.txt', 'w')
    
    # multi_flicker(freqs, duration_sec=60, window_size=(1536, 864))
    multi_flicker(freqs, duration_sec=duration_sec, window_size=window_size, n_repeats=n_repeats, labels=labels, file=file)
    
    file.close()