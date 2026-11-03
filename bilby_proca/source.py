import numpy as np
from numba import njit
from scipy.signal import tukey
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
import pycbc
from pycbc.types import TimeSeries, FrequencySeries
import pickle, os, sys
from bilby.core.utils import solar_mass, \
    parsec, speed_of_light, gravitational_constant
from bilby.gw.utils import lalsim_GetApproximantFromString
from .utils import padding, derivative_to_differencing_FD

from procamiserianew.procamiseria.surrogates import SurrogateModel

surrogate_2220_gaussian = SurrogateModel()

surrogate_2220_gaussian.load('/home/jorgelopez.rodriguez-monteverde/surrogates/procamiseria_unequalmass_discreteloading_32_equal_mass.pkl')

SM_to_m = solar_mass * gravitational_constant / speed_of_light / speed_of_light
a_constant = 4.9254184936691445e-6
scaling = 100 * 1e-6 * SM_to_m / a_constant / a_constant / parsec

def compute_corrected_h2f(h2f, sample_frequencies, delta_t):
    """Apply the correction factor to obtain the Fourier transform of
    second order finite-differenced strain signal from the Fourier transform of
    second order derivative of strain signal.
    
    Parameters
    ----------
    h2f: array-like
        Fourier transform of second order derivative of strain signal.
    sample_frequencies: array-like
        An array of sample frequencies.
    delta_t: float
        Sampling interval in second.
        
    Returns
    -------
    out: array-like
        Fourier transform of second order finite-differenced strain signal.
    """
    h2f = np.array(h2f)
    freq = 2. * np.pi * sample_frequencies * delta_t
    freq2 = freq**2
    freq_idx = freq!=0.
    correction = 2 * (1. - np.cos(freq))
    correction[freq_idx] /= freq2[freq_idx]
    return h2f * correction

@njit()
def mypad(arr, after, before=0):
    length = len(arr) + before + after
    new_arr = np.empty(length, dtype=np.float64)
    new_arr[before : before + len(arr)] = arr
    new_arr[0 : before] = 0
    new_arr[before + len(arr) : ] = 0
    return new_arr

@njit(fastmath = True)
def njitfindiff2(np_arr, delta):
    ''' An (almost) 10 times faster finite differencing method             
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


#Define some quantities

SpeedOfLight = 2.99792458*1e8
GN = 6.674e-11
SunMass = 1.98847*1e30
MegaParsec = 3.08567758128*1e22

if not sys.version_info[1] > 7:
    # Apparently, the pickle is broken for > python 3.7
    # ecc_indir = '/home/juan.calderonbustillo/example/'
    ecc_path = lambda x : os.path.join(os.path.dirname(__file__), "eccentric", x)
    with open(ecc_path('fits_amp.pkl'), 'rb') as f:
        fits_a = pickle.load(f)
    with open(ecc_path('fits_phi.pkl'), 'rb') as f:
        fits_p = pickle.load(f)

    eim_a_B = np.loadtxt(ecc_path('eim_amp_B.txt'))
    eim_p_B = np.loadtxt(ecc_path( 'eim_phi_B.txt'))


## Two Surrogate functions:
 # Evaluate the fits at requested parameter
 # Evaluate the empirical interpolant using the fit evaluations to
 # estimate the BesselJ function at the requested nu parameter

def evalSurrogate_a(param):
    fit_evals = np.array([f(param) for f in fits_a])
    return np.dot(fit_evals, eim_a_B)


def evalSurrogate_p(param):
    fit_evals = np.array([f(param) for f in fits_p])
    return np.dot(fit_evals, eim_p_B)


# Define a few functions first
def rec_phase(frequency,time):
    delta_t = time[1]-time[0]
    my_phase = np.zeros(len(time))
    for j in np.arange(1,len(time)):
        my_phase[j] = my_phase [j-1] + frequency[j]*delta_t

    return my_phase


def Ydirect(iota, phi, l, m):
    Y_lm = lal.SpinWeightedSphericalHarmonic(iota, phi, -2, l, m)
    return Y_lm

def fromNRtoPhysicalUnitsSurrogate(mode, time, times, MD_ratio, delta_t):
    strain = (MD_ratio)* mode

    if len(time) != len(times):
        hinterp = interp1d(time, strain)
        strain = hinterp(times)

    return strain


def fromNRtoPhysicalUnits(mode, time, times, MD_ratio, delta_t):
    strain = (MD_ratio)*(np.transpose(mode)[1]+1j*np.transpose(mode)[2])

    if len(time) != len(times):
        hinterp = interp1d(time, strain)
        strains = hinterp(times)
        return strains
    else:
        return strain


def strain_hcpc_from_dat_file(time_array, total_mass, luminosity_distance,
        theta_jn, phase, geocent_time, strain_paths_set, delta_t, NR_path=""):
    """ A wrapper function to take strain paths as arguments, return with hp & hc.
    It assumes a structure for strain_paths_set being an array of paths,
    consists of at least h22, h20, h32, in this order, and perhaps also h33.
    """
    h22 = np.loadtxt(NR_path+strain_paths_set[0])
    h20 = np.loadtxt(NR_path+strain_paths_set[1])
    h32 = np.loadtxt(NR_path+strain_paths_set[2])
    wfm_args = {"delta_t": delta_t}

    if len(strain_paths_set) == 4:
        h33 = np.loadtxt(NR_path+strain_paths_set[3])
        mylen = np.min([len(h22), len(h20), len(h32), len(h33)])
        wfm_args["h33"] = h33[0:mylen]
    else:
        mylen = np.min([len(h22), len(h20), len(h32)])

    # If any of the strain data is longer than others, shorten it.
    wfm_args.update(dict(h22=h22[0:mylen], h20=h20[0:mylen], h32=h32[0:mylen]))

    return strain_hphc_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **wfm_args)


def psi4_hcpc_from_dat_file(time_array, total_mass, luminosity_distance,
        theta_jn, phase, geocent_time, strain_paths_set, delta_t, NR_path=""):
    """ A wrapper function to take psi4 paths as arguments, return with hp & hc.
    It assumes a structure for strain_paths_set being an array of paths,
    consists of at least h22, h20, h32, in this order, and perhaps also h33.
    """
    h22 = np.loadtxt(NR_path+strain_paths_set[0])
    h20 = np.loadtxt(NR_path+strain_paths_set[1])
    h32 = np.loadtxt(NR_path+strain_paths_set[2])
    wfm_args = {"delta_t": delta_t}

    if len(strain_paths_set) == 4:
        h33 = np.loadtxt(NR_path+strain_paths_set[3])
        mylen = np.min([len(h22), len(h20), len(h32), len(h33)])
        wfm_args["h33"] = h33[0:mylen]
    else:
        mylen = np.min([len(h22), len(h20), len(h32)])

    # If any of the strain data is longer than others, shorten it.
    wfm_args.update(dict(h22=h22[0:mylen], h20=h20[0:mylen], h32=h32[0:mylen]))

    return psi4_hphc_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **wfm_args)


# I have slightly modified the function by including the NR strains as an argument.
def strain_hphc_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):

    h22 = waveform_arguments["h22"]
    h20 = waveform_arguments["h20"]
    h32 = waveform_arguments["h32"]
    h33 = waveform_arguments.get("h33", None)
    mylen = np.min([len(h22),len(h20),len(h32)])
    delta_t = waveform_arguments["delta_t"]

    for h in [h22, h20, h32, h33]:
        if len(h) != mylen:
            try:
                h = h[:mylen]
            except TypeError:
                # If h33 does not exist
                pass
        else:
            pass

    t_start = - total_mass/1300.
    # 1/1300 = 0.2/260
    # This value has be chosen becoz when we scale to 260 Mdot,
    # the entire pre-merger is contained in the first 0.2s, excluding "junk radiation".

    time = total_mass * np.transpose(h22)[0]
    times = np.arange(time[0], time[-1], delta_t)
    MD_ratio = total_mass / luminosity_distance

    h22s = fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h33 is not None:
        h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
        h3m3s = -np.conjugate(h33s)

        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
            Ydirect(theta_jn, phase, 3, -3)*h3m3s

    hp = np.real(H)
    hc = np.imag(H)

    waveform_time = np.arange(0, mylen, delta_t)
    amp22 = np.abs(h22s)

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))

    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index

    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = padding(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = padding(h_cross_windowed, before=N_start, after=N_end)

    return {'plus': h_plus_padded, 'cross': h_cross_padded}


def psi4_hphc_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):

    h22 = waveform_arguments["h22"]
    h20 = waveform_arguments["h20"]
    h32 = waveform_arguments["h32"]
    h33 = waveform_arguments.get("h33", None)
    mylen = np.min([len(h22),len(h20),len(h32)])
    delta_t = waveform_arguments["delta_t"]

    t_start = - total_mass/1300.
    # 1/1300 = 0.2/260
    # 1/2000 is to ensure the junk radiation is completely removed
    # This value has be chosen becoz when we scale to 260 Mdot,
    # the entire pre-merger is contained in the first 0.2s, excluding "junk radiation".

    for h in [h22, h20, h32, h33]:
        if len(h) > mylen:
            try:
                h = h[:mylen]
            except TypeError:
                # If h33 does not exist
                pass
        else:
            pass

    time = total_mass * 4.92e-6 * np.transpose(h22)[0]
    times = np.arange(time[0], time[-1], delta_t)

    MD_ratio = scaling / (total_mass * luminosity_distance)
# It is mass * distance, the original formula was:
# 100./(total_mass*4.9254184936691445e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)
# One can check with def. of `scaling` and simplify the above expression

    h22s = fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h33 is not None:
        h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
        h3m3s = -np.conjugate(h33s)

        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
                Ydirect(theta_jn, phase, 3, -3)*h3m3s

    hp = np.real(H)
    hc = np.imag(H)

    waveform_time = np.arange(0, mylen, delta_t)
    amp22 = np.abs(h22s)

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))

    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index

    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = padding(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = padding(h_cross_windowed, before=N_start, after=N_end)

    original_re_ts = TimeSeries(h_plus_padded, delta_t=delta_t, epoch=time_array[0])
    original_im_ts = TimeSeries(h_cross_padded, delta_t=delta_t, epoch=time_array[0])

    original_re_fs = original_re_ts.to_frequencyseries().data
    original_im_fs = original_im_ts.to_frequencyseries().data

    corrected_re = derivative_to_differencing_FD(
            original_re_fs, original_re_fs.sample_frequencies, delta_t)
    corrected_im = derivative_to_differencing_FD(
            original_im_fs, original_im_fs.sample_frequencies, delta_t)

    dF = original_re_fs.delta_f

    corrected_re_fs = FrequencySeries(corrected_re, dF)
    corrected_im_fs = FrequencySeries(corrected_im, dF)

    corrected_re_ts = corrected_re_fs.to_timeseries().data
    corrected_im_ts = corrected_im_fs.to_timeseries().data

    return {'plus': corrected_re_ts, 'cross': corrected_im_ts}


def strain_hphc_from_ml_eccentric_model(time_array, total_mass, mass_ratio, spin1z, spin2z, 
        eccentricity, luminosity_distance, theta_jn, phase, geocent_time, **waveform_arguments):

    delta_t = time_array[1] - time_array[0]

    ##### Since we know the surrogate does work for 65Msun, let's generate waveforms for this mass and then re-scale
    Mtot_nodim = total_mass / 65
    dL_nodim = luminosity_distance / 100

    mass_1 = 65 * mass_ratio / (1+mass_ratio)
    mass_2 = 65 - mass_1

    scale = Mtot_nodim / dL_nodim
    amp = evalSurrogate_a([mass_1, mass_2, spin1z, spin2z, eccentricity])
    phi = evalSurrogate_p([mass_1, mass_2, spin1z, spin2z, eccentricity])
    h22 = scale * amp * np.exp(1j*phi)

    # The waveform is generated by default at 4096Hz, which may be way higher than what we want. Therefore we resample.
    waveform_true_time_axis = np.arange(0, 1, 1/4096) * Mtot_nodim
    hinterp = interp1d(waveform_true_time_axis, h22)
    # The resampled_time_axis starts with 0, ends with Mtot_nodim * (1 - 1/4096)
    waveform_resampled_time_axis = np.arange(0, Mtot_nodim * 4095/4096, delta_t)
    h22s = hinterp(waveform_resampled_time_axis)

    peak_idx = np.argmax(np.abs(h22s))
    gt_idx = np.argmin(np.abs(time_array - geocent_time))

    ## Retreive as much as possible
    len_diff = peak_idx - gt_idx
    N_start = abs(min(len_diff, 0))
    h22s = h22s[max(len_diff, 0):]
    # Note:
    # if peak_idx > gt_idx: -> too long
    #    h22s = h22s[peak_idx - gt_idx:]
    #    N_start = 0
    # else: -> too short
    #    entire h22
    #    N_start = gt_idx - peak_idx

    ### With pre_peak_len, can replace the above with these:
    # pre_peak_time = waveform_arguments.get('pre_peak_time', 1)
    # if pre_peak_time is not None:
    #     sampling_freq = np.int(1/delta_t)
    #     pre_peak_len = int(pre_peak_time * sampling_freq)
    #     h22s = h22s[peak_idx - pre_peak_len:]
    #     N_start = gt_idx - pre_peak_len

    # Now obtain h2m2
    h2m2s = np.conjugate(h22s)

    H = Ydirect(theta_jn, phase, 2,  2)*h22s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s

    hp = np.real(H)
    hc = np.imag(H)

    # Figure out zero paddings to place the waveform peak at the right place
    time_arr_len = len(time_array)
    wfm_len = len(hp)
    N_end = time_arr_len - wfm_len - N_start

    h_plus = np.zeros(time_arr_len)
    h_cross = np.zeros(time_arr_len)

    window = tukey(time_arr_len - N_start, alpha=0.2)

    h_plus[N_start:-N_end] = hp * window[:wfm_len]
    h_cross[N_start:-N_end] = hc * window[:wfm_len]

    return {'plus': h_plus, 'cross': h_cross}

def strain_hphc_from_ml_eccentric_model_prepeak(time_array, total_mass, mass_ratio, spin1z, spin2z, 
        eccentricity, luminosity_distance, theta_jn, phase, geocent_time, **waveform_arguments):

    delta_t = time_array[1] - time_array[0]

    ##### Since we know the surrogate does work for 65Msun, let's generate waveforms for this mass and then re-scale
    Mtot_nodim = total_mass / 65
    dL_nodim = luminosity_distance / 100

    mass_1 = 65 * mass_ratio / (1+mass_ratio)
    mass_2 = 65 - mass_1

    scale = Mtot_nodim / dL_nodim
    amp = evalSurrogate_a([mass_1, mass_2, spin1z, spin2z, eccentricity])
    phi = evalSurrogate_p([mass_1, mass_2, spin1z, spin2z, eccentricity])
    h22 = scale * amp * np.exp(1j*phi)

    # The waveform is generated by default at 4096Hz, which may be way higher than what we want. Therefore we resample.
    waveform_true_time_axis = np.arange(0, 1, 1/4096) * Mtot_nodim
    hinterp = interp1d(waveform_true_time_axis, h22)
    # The resampled_time_axis starts with 0, ends with Mtot_nodim * (1 - 1/4096)
    waveform_resampled_time_axis = np.arange(0, Mtot_nodim * 4095/4096, delta_t)
    h22s = hinterp(waveform_resampled_time_axis)

    peak_idx = np.argmax(np.abs(h22s))
    gt_idx = np.argmin(np.abs(time_array - geocent_time))

    ## Retreive as much as possible
    # len_diff = peak_idx - gt_idx
    # N_start = abs(min(len_diff, 0))
    # h22s = h22s[max(len_diff, 0):]
    # Note:
    # if peak_idx > gt_idx: -> too long
    #    h22s = h22s[peak_idx - gt_idx:]
    #    N_start = 0
    # else: -> too short
    #    entire h22
    #    N_start = gt_idx - peak_idx

    # With pre_peak_len, can replace the above with these:
    pre_peak_time = waveform_arguments.get('pre_peak_time', 1)
    # if pre_peak_time is not None:
    sampling_freq = np.int(1/delta_t)
    pre_peak_len = int(pre_peak_time * sampling_freq)
    h22s = h22s[peak_idx - pre_peak_len:]
    N_start = gt_idx - pre_peak_len

    # Now obtain h2m2
    h2m2s = np.conjugate(h22s)

    H = Ydirect(theta_jn, phase, 2,  2)*h22s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s

    hp = np.real(H)
    hc = np.imag(H)

    # Figure out zero paddings to place the waveform peak at the right place
    time_arr_len = len(time_array)
    wfm_len = len(hp)
    N_end = time_arr_len - wfm_len - N_start

    h_plus = np.zeros(time_arr_len)
    h_cross = np.zeros(time_arr_len)

    window = tukey(time_arr_len - N_start, alpha=0.2)

    h_plus[N_start:-N_end] = hp * window[:wfm_len]
    h_cross[N_start:-N_end] = hc * window[:wfm_len]

    return {'plus': h_plus, 'cross': h_cross}


def lal_binary_black_hole_time_domain_from_frequency_domain_H0_drift(
        time_array, mass_1, mass_2, luminosity_distance, a_1,tilt_1,phi_12, a_2,tilt_2, phi_jl,
        theta_jn, phase,ra,dec,geocent_time,H0,psi,t_start_template,t_end_template, **kwargs):

    waveform_kwargs = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
                           minimum_frequency=20.0, t_start=0, alpha=0,nr_file = None)
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    alpha = waveform_kwargs['alpha']

    t_start = t_start_template

    if mass_2 > mass_1:
        return None

    luminosity_distance_SI = luminosity_distance * 1e6 * parsec
    mass_1 = mass_1 * solar_mass
    mass_2 = mass_2 * solar_mass

    if tilt_1 == 0 and tilt_2 == 0:
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, reference_frequency, phase)

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None
    delta_time = time_array[1] - time_array[0]

    approximant = lalsim_GetApproximantFromString('IMRPhenomD')

    h_plus_lal, h_cross_lal = lalsim.SimInspiralTD(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance_SI, theta_jn, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_time,
        minimum_frequency, reference_frequency,
        None, approximant)

    amp = np.sqrt(h_plus_lal.data.data**2+h_cross_lal.data.data**2)

    #locate time of peak and match to GPS time
    t1 = np.arange(h_plus_lal.data.length) * h_plus_lal.deltaT
    t1 += np.float(h_plus_lal.epoch)
    max_index = np.argmin(np.abs(t1))

    h_plus_full = h_plus_lal.data.data
    h_cross_full = h_cross_lal.data.data

    amp=np.sqrt(h_plus_lal.data.data**2+h_cross_lal.data.data**2)
    time = np.arange(0,len(h_plus_lal.data.data))*1./h_plus_lal.deltaT
    h_complex = h_plus_lal.data.data-1j*h_cross_lal.data.data
    phase = np.angle(h_complex)
    newphase = np.unwrap(phase)
    freq = np.gradient(newphase,time)

    distance_drift = np.exp(np.arange(0,len(h_plus_lal.data.data))*time*H0*3.24078e-20)
    redshift_term = (1./speed_of_light)*1000*H0*luminosity_distance*distance_drift
    distance_term = 1/(distance_drift)


    rec_phase_with_drift = rec_phase(freq/(1+redshift_term),time)
    rec_amp = amp/(distance_drift)

    new_complex_strain = rec_amp*np.exp(-1j*(rec_phase_with_drift))

    h_plus_full = np.real(new_complex_strain)
    h_plus_cross = np.imag(new_complex_strain)

    #locate time of peak and match to GPS time

    gps_index = np.argmin(np.abs(time_array-geocent_time))

    #locate starting time of the waveform

    start_index_waveform_before_peak = np.max([np.int(t_start/delta_time),-1*gps_index])

    h_plus = h_plus_full[max_index+start_index_waveform_before_peak:]
    h_cross = h_cross_full[max_index+start_index_waveform_before_peak:]

    #Till here, we have retained the piece of waveform we want according to our desired start point. This , however, may be longer than the actual time segment. We will take care of that next.

    if alpha > 0:

        window = tukey(len(h_plus), alpha=alpha)

        start_idx = int(len(window)*0.5)
        end_idx = len(window)
        window[start_idx:end_idx] = 1

    h_plus *= window
    h_cross *= window

    # Determine if we need to add zeros at the begginin

    if gps_index > -1*np.int(t_start/delta_time):
       N_start = gps_index  + np.int(t_start/delta_time)

    else:
       N_start = 0

    # Determine if we need to add zeroes at the end

    h_plus_padded = np.pad(h_plus,(N_start,N_end),'constant')
    h_cross_padded = np.pad(h_cross,(N_start,N_end),'constant')


    return {'plus': h_plus_padded, 'cross': h_cross_padded}

def psi4_hphc_from_surrogate_2220_gaussian_with_correction(time_array, total_mass, omega, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):



    #peak_time = surrogate.get_peak(w, lm = '22')   # Peak in the original files (surrogate is centered on 22 mode)
    t, h22   = surrogate_2220_gaussian(omega, lm = '22')              # Time and psi4 at a given w
    t, h20   = surrogate_2220_gaussian(omega, lm = '20')              # Time and psi4 at a given w

    #time = np.array(time)
    #h22 = np.array(h22)



    mylen = len(h22)
    delta_t = waveform_arguments["delta_t"]

    t_start = - 99
    # 1/1300 = 0.2/260
    # This value has be chosen becoz when we scale to 260 Mdot,
    # the entire pre-merger is contained in the first 0.2s, exlcuding "junk radiation".

    time = total_mass * 4.92e-6 * t
    times = np.arange(time[0], time[-1], delta_t)
    MD_ratio = 100./(total_mass*4.92e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)

 #   print([len(h22),len(time)])

    h22s = fromNRtoPhysicalUnitsSurrogate(h22, time, times, MD_ratio, delta_t)
    h20s = fromNRtoPhysicalUnitsSurrogate(h20, time, times, MD_ratio, delta_t)
    #h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    #h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +  Ydirect(theta_jn, phase, 2, -2)*h2m2s + Ydirect(theta_jn, phase, 2, 0)*h20s
        #Ydirect(theta_jn, phase, 3, 2)*h32s +\
        #Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        #Ydirect(theta_jn, phase, 3, -2)*h3m2s

    #if h33 is not None:
    #    h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
    #    h3m3s = -np.conjugate(h33s)
#
#        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
#            Ydirect(theta_jn, phase, 3, -3)*h3m3s


    hp = np.real(H)
    hc = np.imag(H)

    hp22 = np.real(h22s)
    hc22 = np.imag(h22s)

    waveform_time = np.arange(0, mylen, delta_t)
    amp22 = np.abs(h22s)

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))

    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index

    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = mypad(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = mypad(h_cross_windowed, before=N_start, after=N_end)

    original_re_ts = pycbc.types.TimeSeries(h_plus_padded,delta_t=delta_t,epoch=time_array[0])
    original_im_ts = pycbc.types.TimeSeries(h_cross_padded,delta_t=delta_t,epoch=time_array[0])

    original_re_fs = original_re_ts.to_frequencyseries()
    original_im_fs = original_im_ts.to_frequencyseries()

    corrected_re = compute_corrected_h2f(np.array(original_re_fs), np.array(original_re_fs.sample_frequencies), delta_t)
    corrected_im = compute_corrected_h2f(np.array(original_im_fs), np.array(original_im_fs.sample_frequencies), delta_t)

    dF = original_re_fs.sample_frequencies[1]-original_re_fs.sample_frequencies[0]

    corrected_re_fs = pycbc.types.FrequencySeries(corrected_re,dF)
    corrected_im_fs = pycbc.types.FrequencySeries(corrected_im,dF)


    corrected_re_ts = np.array(corrected_re_fs.to_timeseries())
    corrected_im_ts = np.array(corrected_im_fs.to_timeseries())

    return {'plus': corrected_re_ts, 'cross': corrected_im_ts}

def psi4_hphc_from_mode_arr_with_correction(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):

    h22 = waveform_arguments["h22"]
    h20 = waveform_arguments["h20"]
    h32 = waveform_arguments["h32"]
    h33 = waveform_arguments.get("h33", None)
    mylen = np.min([len(h22),len(h20),len(h32)])
    delta_t = waveform_arguments["delta_t"]

    t_start = - total_mass/1300.
    # 1/1300 = 0.2/260 
    # This value has be chosen becoz when we scale to 260 Mdot, 
    # the entire pre-merger is contained in the first 0.2s, exlcuding "junk radiation".

    time = total_mass * 4.92e-6 * np.transpose(h22)[0]
    times = np.arange(time[0], time[-1], delta_t)
    MD_ratio = 100./(total_mass*4.92e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)

    h22s = fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h33 is not None:
        h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
        h3m3s = -np.conjugate(h33s)

        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
            Ydirect(theta_jn, phase, 3, -3)*h3m3s

    hp = np.real(H)
    hc = np.imag(H)

    hp22 = np.real(h22s)
    hc22 = np.imag(h22s)

    waveform_time = np.arange(0, mylen, delta_t)
    amp22 = np.abs(h22s)

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))

    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index

    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = mypad(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = mypad(h_cross_windowed, before=N_start, after=N_end)

    original_re_ts = pycbc.types.TimeSeries(h_plus_padded,delta_t=delta_t,epoch=time_array[0])
    original_im_ts = pycbc.types.TimeSeries(h_cross_padded,delta_t=delta_t,epoch=time_array[0])

    original_re_fs = original_re_ts.to_frequencyseries()
    original_im_fs = original_im_ts.to_frequencyseries()

    corrected_re = compute_corrected_h2f(np.array(original_re_fs), np.array(original_re_fs.sample_frequencies), delta_t)
    corrected_im = compute_corrected_h2f(np.array(original_im_fs), np.array(original_im_fs.sample_frequencies), delta_t)

    dF = original_re_fs.sample_frequencies[1]-original_re_fs.sample_frequencies[0]

    corrected_re_fs = pycbc.types.FrequencySeries(corrected_re,dF)
    corrected_im_fs = pycbc.types.FrequencySeries(corrected_im,dF)
    

    corrected_re_ts = np.array(corrected_re_fs.to_timeseries())
    corrected_im_ts = np.array(corrected_im_fs.to_timeseries())




    return {'plus': corrected_re_ts, 'cross': corrected_im_ts}

