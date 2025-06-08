import numpy as np
import torch
import threading
import time
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from Models import SCCNet
from scipy.signal import butter, filtfilt, resample_poly
import sys

# ======== Model Setup ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SCCNet(4, 1249, 3).to(device)
checkpoint = torch.load("checkpoints/train-epoch41.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# ======== Globals for Threading ============
BUFFER_SIZE = 1249  # samples needed per prediction
CHANNEL_COUNT = 4  # number of EEG channels
SAMPLE_RATE = 250  # Hz (adjust this based on your Cygnus device) 250
PREDICTION_INTERVAL = 0.2  # seconds between predictions

eeg_buffer = np.zeros((CHANNEL_COUNT, 0))  # initialize empty buffer


# ======== LSL Inlet Setup ============
def setup_lsl_inlet(stream_name="Cygnus-083704-RawEEG"):
    # Option 1: get all streams
    streams = resolve_streams()
    for stream in streams:
        print(stream.name())
    print("Resolving EEG stream...")
    streams = resolve_byprop('name', stream_name)
    inlet = StreamInlet(streams[0])
    print("Stream resolved.")
    return inlet


# ======== EEG Reading Thread ============
def read_eeg(inlet):
    global eeg_buffer
    end = time.time()
    while True:
        sample, _ = inlet.pull_sample()
        # sample = np.array(sample[:CHANNEL_COUNT]).reshape(-1, 1)  # shape: (4,1)
        # print(f"sample[0:2] + sample[4:6]: {sample[0:2] + sample[4:6]}")
        sample = np.array(sample[0:2] + sample[4:6]).reshape(-1, 1)  # shape: (4,1)
        eeg_buffer = np.hstack((eeg_buffer, sample))
        # start = time.time()
        # print(f"eeg_buffer.shape[1]: {eeg_buffer.shape[1]}")
        if eeg_buffer.shape[1] > BUFFER_SIZE:
            eeg_buffer = eeg_buffer[:, -BUFFER_SIZE:]  # keep last BUFFER_SIZE samples
            # print(f"last eeg_buffer data: {eeg_buffer[:, BUFFER_SIZE - 1]}")  # equal to sample[0:2] + sample[4:6]
            # print(f"eeg_buffer read time: {start-end}")
        time.sleep(1.0 / SAMPLE_RATE)


# ======== Prediction Thread ============
def predict_loop():
    global eeg_buffer
    while True:
        if eeg_buffer.shape[1] >= BUFFER_SIZE:
            input_data = eeg_buffer[:, -BUFFER_SIZE:].copy()  # shape:
            input_data = np.transpose(input_data, (1, 0))
            b, a = butter(4, [1 / (0.5 * SAMPLE_RATE), 20 / (0.5 * SAMPLE_RATE)], btype='band')
            input_data = filtfilt(b, a, input_data, axis=0)
            input_data = np.transpose(input_data, (1, 0))
            # print(input_data.shape)
            # print(input_data)
            x_batch = torch.from_numpy(input_data.copy()).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,4,1249)
            with torch.no_grad():
                outputs = model(x_batch)
                prediction = int(torch.argmax(outputs).cpu().item())
                print(f"Prediction: {prediction}")
        time.sleep(PREDICTION_INTERVAL)


def ssvep_window():
    from psychopy import visual, event
    import numpy as np

    win = visual.Window(size=(800, 800), color='black', units='pix', fullscr=False)
    frequencies = [4, 8, 15]
    size = 300
    frame_rate = win.getActualFrameRate(nIdentical=60, nMaxFrames=120, nWarmUpFrames=60)
    # frame_rate = win.getActualFrameRate()
    if frame_rate is None:
        frame_rate = 60.0  # 預設為 60 Hz
    print(f"Actual Frame Rate: {frame_rate:.2f} Hz")
    stimuli = []
    stimuli.append(visual.Rect(win, width=size * 2, height=size * 2, pos=(0, 0), fillColor='red', lineColor='red'))
    stimuli.append(visual.Circle(win, radius=size, pos=(0, 0), fillColor='green', lineColor='green'))
    triangle_vertices = [(0, size), (-size, -size), (size, -size)]
    stimuli.append(visual.ShapeStim(win, vertices=triangle_vertices, pos=(0, 0), fillColor='blue', lineColor='blue'))
    index = 0
    while True:
        # === 根據頻率產生 frame 閃爍序列 ===
        n_frames = int(frame_rate)
        frames_per_cycle = frame_rate / frequencies[index]
        pattern = np.array([(i % frames_per_cycle) < (frames_per_cycle / 2) for i in range(n_frames)])

        # === 開始呈現刺激 ===
        for i in range(n_frames):
            stimuli[index].opacity = 1.0 if pattern[i] else 0.0
            stimuli[index].draw()
            win.flip()

        key_list = event.getKeys()
        if 'escape' in key_list:
            break
        if 'a' in key_list:
            index -= 1
            if index < 0:
                index = len(frequencies) - 1
        if 'd' in key_list:
            index += 1
            if index > len(frequencies) - 1:
                index = 0

    win.close()


# ======== Main ============
def main():
    inlet = setup_lsl_inlet()
    #
    thread1 = threading.Thread(target=read_eeg, args=(inlet,), daemon=True)
    thread2 = threading.Thread(target=predict_loop, daemon=True)

    thread1.start()
    thread2.start()

    # print("Running... Press Ctrl+C to exit.")
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Stopped.")
    ssvep_window()  # run directly, not via threading


if __name__ == "__main__":
    main()
