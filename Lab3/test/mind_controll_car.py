from pylsl import StreamInlet, resolve_streams, resolve_byprop
from pylsl import StreamInlet
import numpy as np
# import numpy as np
import serial
import queue
import time
import threading
from scipy.signal import butter, filtfilt, resample_poly
SAMPLE_RATE = 1000  # Hz (adjust this based on your Cygnus device) 25
thres = 0.0  # threshold
# qsize = 30  # max queue size
CHANNEL_COUNT = 4  # number of EEG channels
BUFFER_SIZE = 500  # samples needed per prediction
eeg_buffer = np.zeros((CHANNEL_COUNT, BUFFER_SIZE))


# ======== EEG Reading Thread ============
def read_eeg(inlet):
    global eeg_buffer
    end = time.time()
    print("@@@@@@@@@@@@@@@@@@@@@")
    while True:
        sample, _ = inlet.pull_sample()
        sample = np.array(sample[0:2] + sample[4:6]).reshape(-1, 1)  # shape: (4,1)
        eeg_buffer[:, :-1] = eeg_buffer[:, 1:]  # 向左移動一格，丟掉最舊的
        eeg_buffer[:, -1] = sample.flatten()
        # print(f"eeg_buffer: {eeg_buffer.shape}")
        time.sleep(1.0 / SAMPLE_RATE)


def main():
    import argparse
    #parser = argparse.ArgumentParser(description="CECNL BCI 2023 Car Demo")
    #parser.add_argument("port_num", type=str, help="Arduino bluetooth serial port")
    #args = parser.parse_args()

    # ser = serial.Serial(args.port_num, 9600, timeout=1, write_timeout=1)

    # q = queue.Queue(maxsize=qsize)

    # streams = resolve_stream('name', 'OpenViBE Stream1')
    # create a new inlet to read from the stream
    # inlet = StreamInlet(streams[0])

    b, a = butter(1, [1 / (0.5 * SAMPLE_RATE), 40 / (0.5 * SAMPLE_RATE)], btype='band')

    while True:
        # 檢查是否資料已經有意義（非零）可過濾
        if np.count_nonzero(eeg_buffer) < BUFFER_SIZE * CHANNEL_COUNT:
            print("Waiting for buffer to fill...")
            time.sleep(0.2)
            continue

        # Bandpass filter
        input_data = filtfilt(b, a, eeg_buffer, axis=1)  # ← 注意 axis=1，因為每列是 channel

        ratio = np.sum(input_data[0]) / BUFFER_SIZE

        if ratio > thres:
            print("move forward", ratio)
            # ser.write(b'1')
        else:
            print("stop ", ratio)
            # ser.write(b'0')

        time.sleep(0.2)


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


if __name__ == "__main__":
    inlet = setup_lsl_inlet()
    thread1 = threading.Thread(target=read_eeg, args=(inlet,), daemon=True)

    thread1.start()
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)
