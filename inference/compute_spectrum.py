
import numpy as np
import scipy.stats as stats

def compute_spectrum2d(data):
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, N_y, N_x = data.shape
    if N_x == 2 * N_y:
        data1 = data[:, :, :N_y]
        data2 = data[:, :, N_y:]
        data = np.concatenate((data1, data2), axis=0)
    N, N_y, N_x = data.shape

    # Take FFT and take amplitude
    fourier_image = np.fft.fftn(data, axes=(1,2))
    fourier_amplitudes = np.abs(fourier_image)**2

    # Get kx and ky
    kfreq_x = np.fft.fftfreq(N_x) * N_x
    kfreq_y = np.fft.fftfreq(N_y) * N_y

    # Combine into one wavenumber for both directions
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = np.repeat(knrm[np.newaxis, ...], repeats=N, axis=0)

    # Flatten arrays
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Get k-bins and mean amplitude within each bin
    kbins = np.arange(0.5, N_x//2, 1)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)

    # Multiply by volume of bin
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    # Get center of k-bin for plotting
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    return (kvals, Abins)