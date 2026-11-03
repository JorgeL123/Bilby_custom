#! /usr/bin/env python3
# If it does not work on your env, you may try to first source this env:
# source /home/samson.leong/miniconda3/bin/activate; conda activate --stack igwn-py37
import numpy as np
from numba import njit, prange
from math import sqrt as msqrt
from scipy.interpolate import interp1d

import lal

import logging
logger = logging.getLogger('Bilby-Extra')

## Math utils
def infft(frequency_domain_strain, sampling_frequency):
    # Copied from Bilby.core.utils
    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain

def infft_from_ifo(ifo):
    td_waveform = 1 * infft(ifo.whitened_frequency_domain_strain,
            sampling_frequency=ifo.strain_data.sampling_frequency)\
                    / (4*np.pi*ifo.duration)
    return td_waveform


@njit()
def padding(arr, after, before=0):
    length = len(arr) + before + after
    new_arr = np.empty(length, dtype=np.float64)
    new_arr[before : before + len(arr)] = arr
    new_arr[0 : before] = 0
    new_arr[before + len(arr) : ] = 0
    return new_arr


@njit()
def rolling(arr, shift):
    result = np.zeros(len(arr), dtype=arr.dtype)
    result[shift:] = arr[:-shift]
    result[:shift] = arr[-shift:]
    return result


@njit(fastmath=True)
def inner_prod_ax1(a, b):
    inner = np.empty(a.shape[0], dtype=a.dtype)
    for o_idx in range(a.shape[0]):
        sum_ = 0
        for i_idx in range(a.shape[1]):
            sum_ += a[o_idx, i_idx] * b[o_idx, i_idx]
        inner[o_idx] = sum_
    return inner


@njit(fastmath=True)
def snr(num, s_sr, shift_t_sr):
    rho = 0.
    for idx in prange(len(num)):
        rho += num[idx] / msqrt(s_sr[idx] * shift_t_sr[idx])
    return rho


def max_overlap(strain_ifos, template_ifos, duration):
    # extract the whitened frequency strain from the ifos
    strain = np.empty(
        (len(strain_ifos), len(strain_ifos[0].whitened_frequency_domain_strain)),
        dtype=np.complex64)
    template = np.empty(
        (len(template_ifos), len(template_ifos[0].whitened_frequency_domain_strain)),
        dtype=np.complex64)
    for idx in range(len(strain_ifos)):
        strain[idx] = strain_ifos[idx].whitened_frequency_domain_strain
        template[idx] = template_ifos[idx].whitened_frequency_domain_strain

    sampling_freq = template_ifos[0].strain_data.sampling_frequency

    # Interpolate the strain if it has different sampling rate
    if strain.size != template.size:
        strain_interp = interp1d(strain_ifos[0].frequency_array, strain, axis=1, bounds_error=False, fill_value=0)
        freq_array = template_ifos[0].frequency_array
        strain = strain_interp(freq_array)

    # FFT freq_domain_strain -> time_domain_strain
    s_TD = -1. * infft(strain, sampling_frequency=sampling_freq)
    t_TD_fix = -1. * infft(template, sampling_frequency=sampling_freq)

    # Compute the norm of the strain first
    s_sr = inner_prod_ax1(s_TD, s_TD)

    offset, old_rho = 0., 0.
    total = int(duration/0.0002)
    for idx in range(0, total):
        delta = - (duration / 2) + (idx / total)
        # Cut the shifted tail to the head
        t_TD = np.roll(t_TD_fix, int(delta * sampling_freq), axis=1)

        shift_t_sr = inner_prod_ax1(t_TD, t_TD)
        numerator = inner_prod_ax1(s_TD, t_TD)
        # Sum over the detectors
        rho = snr(numerator, s_sr, shift_t_sr)

        # Update the rho and offset when a higher rho appears
        if rho > old_rho:
            old_rho = rho
            offset = delta

    return old_rho, offset


@njit(fastmath = True)
def njit_2nd_finite_diff(np_arr, delta):
    ''' An (almost) 10 times faster (second order) finite differencing method
    (with numba njit, it can go up to 50 times faster)
    Compared to this:
        x = np.array(x)
        N = x.shape[0]
        return (np.roll(x, -1) - 2 * x + np.roll(x, 1))/delta_t**2
    '''
    leng = len(np_arr)
    result = np.zeros(leng)
    result[0] = np_arr[-1] + np_arr[1] - np_arr[0] * 2
    result[-1] = np_arr[-2] + np_arr[0] - np_arr[-1] * 2
    result[1:-1] = np_arr[:-2] + np_arr[2:] - 2*np_arr[1:-1]
    return result / (delta ** 2)


def Psi4_PSD(strain_psd, frequencies, dt):
    """Compute the new PSD of noise after the second order finite differencing.

    Parameters
    ----------
    psd: array-like
        Original PSD.
    sample_frequencies: array-like
        Sample frequencies in Hz.
    delta_t: float
        Time resolution of time series in second.

    Returns
    -------
    out: 1D numpy array
        The new PSD of noise after the second order finite differencing.
    """
    dt2 = dt * dt
    dt4 = dt2 * dt2
    fdt = 2 * np.pi * frequencies * dt
    numerator = 6 - 8 * np.cos(fdt) + 2 * np.cos(2 * fdt)
    return numerator * strain_psd / dt4


def derivative_to_differencing_FD(derivative, frequencies, delta_t):
    """
    Apply the correction factor to obtain the Fourier transform of
    second order finite-differenced strain signal from the Fourier transform of
    second derivative of strain signal.

    Parameters
    ----------
    derivative: array-like
        Fourier transform of second order derivative of strain signal.
    frequencies: array-like
        An array of sample frequencies.
    delta_t: float
        Sampling interval in second.

    Returns
    -------
    out: array-like
        Fourier transform of second order finite-differenced strain signal.
    """
    phase = 2 * np.pi * frequencies * delta_t
    non_zero_idx = phase != 0
    non_zero_phase = phase[non_zero_idx]

    correction = np.zeros(len(phase))
    correction[non_zero_idx] = \
        2 * (1 - np.cos(non_zero_phase)) / non_zero_phase / non_zero_phase

    return derivative * correction


def Ydirect(iota, phi, l, m):
    Y_lm = lal.SpinWeightedSphericalHarmonic(iota, phi, -2, l, m)
    return Y_lm
