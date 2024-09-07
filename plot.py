import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

ratios : np.ndarray = np.load('./fg-ratio.npy')

plt.figure(dpi=120, figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(ratios, '-o')
plt.grid(True)
plt.title(f'min {ratios.min()}, max {ratios.max()}')

plt.subplot(1, 3, 2)
# 每个点相距 100 ms，普通人的反应速度为 500ms，做两倍容错，为 1100ms
kernel = np.hanning(11)
conv_result = convolve(ratios, kernel, mode='same')
plt.plot(conv_result, '-o')
plt.grid(True)
plt.title(f'After 1D Conv Filter, max {conv_result.max()}')

def schmidt_rectification(data, low_threshold, high_threshold):
    output = np.zeros_like(data)
    state = 0
    for i in range(len(data)):
        if state == 0:
            if data[i] > high_threshold:
                state = 1
                output[i] = 1
            else:
                output[i] = 0
        elif state == 1:
            if data[i] < low_threshold:
                state = 0
                output[i] = 0
            else:
                output[i] = 1
    return output

plt.subplot(1, 3, 3)
sch_result = schmidt_rectification(conv_result, 0.2, 0.45)
plt.plot(sch_result)
plt.grid(True)
plt.title(f'After Schmidt Filter')

plt.savefig('plot.png')