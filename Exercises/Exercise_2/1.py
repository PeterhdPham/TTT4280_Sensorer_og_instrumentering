import numpy as np
import matplotlib.pyplot as plt

# Given values
amplitude = 1  # Amplitude of the sine wave in volts
frequency = 100  # Frequency of the sine wave in Hz
sampling_interval = 0.2e-3  # Sampling interval in seconds (0.2 ms)

# Calculate the sampling frequency
sampling_frequency = 1 / sampling_interval  # Sampling frequency in Hz

# Nyquist frequency
nyquist_frequency = sampling_frequency / 2

# Time vector for 900 points
t = np.arange(0, 900 * sampling_interval, sampling_interval)

# Generate the sine wave
x_t = amplitude * np.sin(2 * np.pi * frequency * t)

# Plot the first 200 points
plt.figure(figsize=(12, 6))
plt.plot(t[:200], x_t[:200])
plt.title('Sine Wave Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)
plt.show()

print(sampling_frequency, nyquist_frequency)


# Number of points for FFT
NFFT = 1024

# Since we need 1024 data points for the FFT, we'll truncate or zero-pad the x_t signal as necessary
# We will use the same time interval as before to remain consistent with the original signal

# Zero-padding the signal if necessary
if len(x_t) < NFFT:
    x_t = np.pad(x_t, (0, NFFT - len(x_t)), 'constant')
elif len(x_t) > NFFT:
    x_t = x_t[:NFFT]

# Perform FFT
fft_x_t = np.fft.fft(x_t)

# Create frequency axis and calculate frequency interval Δf
# The frequency interval Δf is given by the sampling frequency divided by the number of FFT points
freq_axis = np.fft.fftfreq(NFFT, d=sampling_interval)
delta_f = sampling_frequency / NFFT

# Only take the positive frequencies (first half of the spectrum)
positive_freq_axis = freq_axis[:NFFT//2]
positive_fft_x_t = fft_x_t[:NFFT//2]

# Plotting the magnitude of the FFT
plt.figure(figsize=(12, 6))
plt.plot(positive_freq_axis, np.abs(positive_fft_x_t))
plt.title('Magnitude Spectrum of the Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

print(delta_f)


# Compute power spectral density (PSD)
psd_x_t = np.abs(fft_x_t)**2

# Plotting the PSD up to 5 kHz
# We first need to find the index corresponding to 5 kHz
index_5kHz = np.where(positive_freq_axis <= 5000)[0][-1]

plt.figure(figsize=(12, 6))
plt.plot(positive_freq_axis[:index_5kHz], psd_x_t[:index_5kHz])
plt.title('Power Spectral Density (PSD) of the Sine Wave up to 5 kHz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True)
plt.show()

# Identify the frequency of the peak (alias) within the 5 kHz range
alias_frequency = positive_freq_axis[np.argmax(psd_x_t[:index_5kHz])]

# Zooming in on the spectrum between 0 Hz and 200 Hz
# We first need to find the index corresponding to 200 Hz
index_200Hz = np.where(positive_freq_axis <= 200)[0][-1]

plt.figure(figsize=(12, 6))
plt.plot(positive_freq_axis[:index_200Hz], psd_x_t[:index_200Hz])
plt.title('Zoomed Power Spectral Density (PSD) of the Sine Wave between 0 Hz and 200 Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True)
plt.show()

print(alias_frequency, np.max(psd_x_t[:index_200Hz]))


# Convert PSD to decibels
psd_in_dB = 20 * np.log10(np.abs(fft_x_t))

# Normalize the spectrum to make the maximum 0 dB
psd_in_dB_normalized = psd_in_dB - np.max(psd_in_dB)

# Plot the normalized PSD in dB
plt.figure(figsize=(12, 6))
plt.plot(positive_freq_axis, psd_in_dB_normalized[:NFFT//2])
plt.title('Normalized Power Spectral Density (PSD) in dB')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power, [dB]')
plt.grid(True)
# Set y-axis to cover the range from -80 dB to 10 dB
plt.ylim(-80, 10)
plt.show()


# Apply a Hanning window to the signal
hanning_window = np.hanning(NFFT)
windowed_signal = x_t * hanning_window

# Perform FFT on the windowed signal
fft_windowed_signal = np.fft.fft(windowed_signal)

# Compute power spectral density (PSD) for both the original and windowed signals
psd_original = np.abs(fft_x_t)**2
psd_windowed = np.abs(fft_windowed_signal)**2

# Convert PSD to decibels
psd_original_dB = 10 * np.log10(psd_original)
psd_windowed_dB = 10 * np.log10(psd_windowed)

# Plotting the PSD in dB with and without the Hanning window
plt.figure(figsize=(12, 6))
plt.plot(positive_freq_axis, psd_original_dB[:NFFT//2], label='Without Hanning Window')
plt.plot(positive_freq_axis, psd_windowed_dB[:NFFT//2], label='With Hanning Window')
plt.title('Power Spectral Density (PSD) with and without Hanning Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density [dB]')
plt.legend()
plt.grid(True)
plt.show()


t_complex = np.arange(0, NFFT * sampling_interval, sampling_interval)
omega = 2 * np.pi * frequency
complex_sine_wave = np.exp(-1j * omega * t_complex)
fft_complex_sine_wave = np.fft.fft(complex_sine_wave)
psd_complex_sine_wave = np.abs(fft_complex_sine_wave)**2
psd_complex_sine_wave_dB = 10 * np.log10(psd_complex_sine_wave)
freq_axis_complex = np.fft.fftfreq(NFFT, d=sampling_interval)

# Plotting the PSD in dB for the complex sine wave up to the sampling frequency fs
plt.figure(figsize=(12, 6))
plt.plot(freq_axis_complex[:NFFT//2], psd_complex_sine_wave_dB[:NFFT//2], label='Complex Sine Wave')
plt.title('Power Spectral Density (PSD) of the Complex Sine Wave up to fs')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density [dB]')
plt.grid(True)
plt.show()


fft_complex_shifted = np.fft.fftshift(fft_complex_sine_wave)
psd_complex_shifted = np.abs(fft_complex_shifted)**2
psd_complex_shifted_dB = 10 * np.log10(psd_complex_shifted)


freq_axis_shifted = np.fft.fftshift(freq_axis_complex)

# Plot the shifted PSD
plt.figure(figsize=(12, 6))
plt.plot(freq_axis_shifted, psd_complex_shifted_dB, label='Shifted Complex Sine Wave')
plt.title('Shifted Power Spectral Density (PSD) of the Complex Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density [dB]')
plt.grid(True)
plt.show()

max_freq_index = np.argmax(psd_complex_shifted_dB)
max_freq_at_zero = freq_axis_shifted[max_freq_index]
print(max_freq_at_zero)
new_complex_sine_wave = np.exp(1j * omega * t_complex)
fft_new_complex_sine_wave = np.fft.fft(new_complex_sine_wave)
fft_new_complex_shifted = np.fft.fftshift(fft_new_complex_sine_wave)
psd_new_complex_shifted = np.abs(fft_new_complex_shifted)**2
psd_new_complex_shifted_dB = 10 * np.log10(psd_new_complex_shifted)

# Plot the shifted PSD of the new complex sine wave
plt.figure(figsize=(12, 6))
plt.plot(freq_axis_shifted, psd_new_complex_shifted_dB, label='Shifted New Complex Sine Wave')
plt.title('Shifted Power Spectral Density (PSD) of the New Complex Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density [dB]')
plt.grid(True)
plt.show()

# Find the index of the maximum value of the new spectrum
max_freq_index_new = np.argmax(psd_new_complex_shifted_dB)
max_freq_new = freq_axis_shifted[max_freq_index_new]

print(max_freq_new)


