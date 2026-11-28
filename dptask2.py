import numpy as np
import matplotlib.pyplot as plt
import pywt
sampling_rate = 1000
time = np.linspace(0, 1, sampling_rate)
signal = np.sin(2 * np.pi * 10 * time)
signal[500:] += np.sin(2 * np.pi * 50 * time[500:])
noise = 0.2 * np.random.randn(len(time))
signal = signal + noise
wavelet = 'db4'
coefficients = pywt.wavedec(signal, wavelet, level=4)
approximation = coefficients[0]
details = coefficients[1:]
scales = np.arange(1, 128)

cwt_coefficients, frequencies = pywt.cwt(
    signal,
    scales,
    'morl',
    sampling_period=1 / sampling_rate
)
power = np.abs(cwt_coefficients)
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(time, signal)
plt.title('Original Signal')
plt.xlabel('Time (Seconds)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(approximation)
plt.title('DWT - Approximation Coefficients')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(3, 1, 3)
plt.imshow(
    power,
    extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
    aspect='auto',
    cmap='jet'
)
plt.title('Continuous Wavelet Transform (CWT)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Wavelet Power')

plt.tight_layout()
plt.show()

print("Wavelet Transform Completed Successfully!")
print("Wavelet Type:", wavelet)
print("Decomposition Levels:", len(details))
print("Signal Length:", len(signal))