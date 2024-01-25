import numpy as np
import matplotlib.pyplot as plt

# Given values
amplitude = 1  # in volts
frequency = 100  # in hertz
delta_t_ms = 0.2  # sampling time in milliseconds
delta_t = delta_t_ms / 1000  # convert milliseconds to seconds
N_FFT = 1024 # number of datapoints for FFT
# Calculate the sampling frequency (fs)
sampling_frequency = 1 / delta_t


# Time vector for 900 points
t = np.linspace(0, 900 * delta_t, 900)

# Generate the sinusoidal signal x(t)
x_t = amplitude * np.sin(2 * np.pi * frequency * t)

# # Plot the first 200 data points
# plt.figure(figsize=(10, 4))
# plt.plot(t[:200], x_t[:200], label='Sinusoidal Signal')
# plt.title('Sinusoidal Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (V)')
# plt.grid(True)
# plt.legend()
# plt.show()

# Generate the sinusoidal signal x(t) with 1024 data points
t_fft = np.linspace(0, N_FFT * delta_t, N_FFT, endpoint=False)
x_t_fft = amplitude * np.sin(2 * np.pi * frequency * t_fft)

# Perform the FFT on the signal
fft_result = np.fft.fft(x_t_fft)

# Generate the frequency axis for the FFT and calculate the frequency interval Delta f
freq_axis = np.fft.fftfreq(N_FFT, delta_t)
delta_f = freq_axis[1] - freq_axis[0]

# calculate the  spectral density spectrum

s_x_f = (np.abs(fft_result)**2)

# For visualization purposes, we'll plot the magnitude of the FFT result
# We only need to plot the first half of the FFT result due to symmetry
# plt.figure(figsize=(10, 4))
# plt.plot(freq_axis[:N_FFT // 2], np.abs(fft_result[:N_FFT // 2]))
# plt.title('Magnitude of FFT of the Signal')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 4))
plt.plot(freq_axis[:N_FFT], s_x_f)
plt.title('Effect spectral density S_X')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


