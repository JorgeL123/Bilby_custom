import numpy as np
from scipy.signal import tukey
from scipy.interpolate import interp1d
import lalsimulation as lalsim
from pycbc.types import TimeSeries, FrequencySeries
from bilby.gw.utils import lalsim_GetApproximantFromString
from ..utils.compute_utils import padding, derivative_to_differencing_FD, Ydirect, rolling
from lal import MSUN_SI, MTSUN_SI, MRSUN_SI, C_SI, PC_SI


### Replacing certain constants defined here with LAL defintions:
## This `SM_to_m` is the Geometrized solar mass, MRSUN_SI.
# SM_to_m = solar_mass * gravitational_constant / speed_of_light / speed_of_light
# => SM_to_m = MRSUN_SI
## This `a_constant` looks like the MTSUN_SI from LAL, replacing it
# a_constant = 4.9254184936691445e-6
# => a_constant = MTSUN_SI
# scaling = 100 * 1e-6 * SM_to_m / a_constant / a_constant / parsec
# => scaling = 100 * 1e-6 * MRSUN_SI / MTSUN_SI / MTSUN_SI / parsec
## Since MRSUN_SI / MTSUN_SI is just C_SI, replacing it:
scaling = 100 * 1e-6 * C_SI / MTSUN_SI / PC_SI

# Using tuple to make sure it is immutable, also as a key in dictionary
PROCA_MODES = [(2,2), (2,0), (3,2), (3,3)]

# Define a few functions first
def rec_phase(frequency,time):
    delta_t = time[1]-time[0]
    my_phase = np.zeros(len(time))
    for j in np.arange(1,len(time)):
        my_phase[j] = my_phase [j-1] + frequency[j]*delta_t

    return my_phase


def deprecated_fromNRtoPhysicalUnits(mode, time, times, MD_ratio, delta_t):
    strain = MD_ratio * (mode.T[1] + 1j*mode.T[2])

    if len(time) != len(times):
        hinterp = interp1d(time, strain)
        strains = hinterp(times)
        return strains
    else:
        return strain


# Update on 09/11/22, remove dummy `delta_t` variable
def fromNRtoPhysicalUnits(mode:np.ndarray, time:np.ndarray,
                          times:np.ndarray, MD_ratio:float):
    strain = MD_ratio * (mode.T[1] + 1j*mode.T[2])

    if len(time) != len(times):
        hinterp = interp1d(time, strain)
        return hinterp(times)
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

    h22s = deprecated_fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = deprecated_fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = deprecated_fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h33 is not None:
        h33s = deprecated_fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
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


def psi4_hphc_from_generic_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):
    """
    Generate Psi4 waveform from the given mode arrays (within waveform_arguments).

    The architecture of this model is to generate a waveform segment from the given
    NR mode arrays, and then pads with zeroes before and after it to match its length
    with the given time array.

    Important arguments in the `waveform_arguments`:
    1. "mode_array" -- list of modes (l, m) to be used in the waveform.
                        Note that in case of (l, -m) not listed, it assumes
                        the aligned spin case, and the (l, -m) mode is simply the
                        complex conjugate of the (l, m) mode (and the sign depends
                        on the oddness of l).
                        Default: PROCA_MODES = [(2,2), (2,0), (3,2), (3,3)].
    2. "hlm" -- They should be the mode arrays coming from NR simulations.
                The expected shape of them is (length, 3), i.e.: 3 long columns;
                in the format of (time, h+, h×). Examples of the "keys" can be like
                "h22" -> (2,2), "h3-2" -> (3,-2), in general, simply "h{l}{m}".
                Note that `mode_array` dominates the final modes present,
                regardless of the number of provided hlm.
                Also, "h22" must be present to define `geocent_time`.
    3. "gt0" -- If true, disregard the input `geocent_time` and set it to `start_time`
                of the `time_array`. This is to align with the convension in Bilby,
                time shift is only handled during injection.
                Default: True.
    4. "pre_merger" -- in sec. Time retain before merger, actual duration scales by
                       total_mass. Specifically:
                            t_start = -pre_merger * total_mass / 260
    5. "max_post_merger" -- in sec. The maximal post merger duration, default is None.
                            If None, shorter of:
                                - remainging time after `geocent_time`, and
                                - tail of waveform segment
                            otherwise, shorter of:
                                - maximal post merger duration, and
                                - tail of waveform segment


    ** Important notes **
    1. The `geocent_time` is designed to match with the moment when the (2, 2) mode
        is at its maximum; however, since this is a Psi4 model, such defintion does not
        match exactly to the conventional `geocent_time` defined with strain.
    2. The tail of the waveform segment is chosen to be the shorter of:
        - the raw output from NR modes, or
        - the remaining time between `geocent_time` and the end of the `time_array`
        Hence, if the given `geocent_time` is outside of the `time_array`, this will
        effectively trim off the ringdown and part of the merger.
    3. "Padding" is achieved by first placing the waveform segment at the beginning of
        the `time_array` and then rolls it to match the (2,2) peak with the `geocent_time`.
        If the `geocent_time` is too close to the `start_time` of the `time_array`, then
        part of the pre-merger phase would be "rolled" to the end of the output waveform.
        This has been tested and would not intoduce artefacts during FFT.
    """

    # print(f"{time_array[0] = }, {time_array[-1] = }, {geocent_time = }")
    delta_t = time_array[1] - time_array[0]
    max_post_merger = waveform_arguments.get('max_post_merger', None)
    pre_merger_length = waveform_arguments.get('pre_merger', 0.2)
    t_start = - total_mass * pre_merger_length / 260.
    # 1/1300 = 0.2/260
    # 1/2000 is to ensure the junk radiation is completely removed
    # This value has be chosen becoz when we scale to 260 Mdot,
    # the entire pre-merger is contained in the first 0.2s, excluding "junk radiation".

    # Rescale the time array to physical unit, the resample it to given delta_t
    h22_time = total_mass * 4.92e-6 * waveform_arguments["h22"].T[0]
    resampled_wfm_time = np.arange(h22_time[0], h22_time[-1], delta_t)
    H = np.zeros(len(resampled_wfm_time), dtype=complex)      # initialise waveform

    MD_ratio = scaling / (total_mass * luminosity_distance)
    # It is mass * distance, the original formula was:
    # 100./(total_mass*4.9254184936691445e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)
    # One can check with def. of `scaling` and simplify the above expression

    # The main reason to include this mode_array is
    # because I don't want to decode the dictionary keys to get (l, m)
    # It also has the benefits of controlling the modes without messing with the mode dictionary
    mode_array = waveform_arguments.get('mode_array', PROCA_MODES)
    for mode in mode_array:
        # Looping over each given modes
        l, m = mode
        key = f'h{l}{m}'
        h_mode = waveform_arguments.get(key, None)

        if h_mode is None:
            continue

        scaled_wfm_time = total_mass * 4.92e-6 * h_mode.T[0]
        resample_h_mode = fromNRtoPhysicalUnits(
            h_mode, scaled_wfm_time, resampled_wfm_time, MD_ratio)
        H += Ydirect(theta_jn, phase, l, m) * resample_h_mode

        if ( m != 0 ) and ( (l, -m) not in mode_array ):
            resample_h_mode_conj = (-1)**l * np.conjugate(resample_h_mode)
            H += Ydirect(theta_jn, phase, l, -m) * resample_h_mode_conj
        if tuple(mode) == (2, 2):
            template_max_index = np.argmax(np.abs(resample_h_mode))

    template_max_time = resampled_wfm_time[template_max_index]
    template_start_idx = np.argmin(np.abs(resampled_wfm_time - (template_max_time + t_start)))

    # Determine length of waveform segment to retain
    gps_index = np.argmin(np.abs(time_array - geocent_time))
    if max_post_merger is not None:
        tmp_end_idx = np.argmin(np.abs(resampled_wfm_time - (template_max_time + max_post_merger)))
        # plus 1 index to match when max_post_merger is longer than the waveform tail
        template_end_idx = min(tmp_end_idx + 1, len(H))
    else:
        # Take the shorter of the two:
        # Time_array tail length: len(time_array) - gps_index
        # Waveform tail length:   len(H) - template_max_index
        template_end_idx = min(len(time_array) - gps_index + template_max_index, len(H))
    template_length = template_end_idx - template_start_idx
    # peak_to_start_len = template_max_index - template_start_idx

    # Initialise the final output waveform
    ta_len = len(time_array)
    H_output = np.zeros(ta_len, dtype=complex)
    # Place the waveform segment at the beginning of the output array
    H_output[:template_length] = H[template_start_idx : template_end_idx]
    # Apply tukey window to 1/3 of pre merger
    duration = time_array[-1] - time_array[0]
    alpha = (template_max_index - template_start_idx) * delta_t / duration * (2/3)
    # (max - start) * delta_t / 2 / (time_array[-1] / 2)
    window = tukey(ta_len, alpha=alpha)
    H_output = H_output * window

    # (temp_max_index - temp_start_idx) is the new peak idx after trimming the starting segment
    gt0 = waveform_arguments.get('gt0', True)
    gps_idx = 0 if gt0 else gps_index
    shift =  gps_idx - (template_max_index - template_start_idx)
    # Shift the waveform to the appropriate position
    if shift != 0:
        H_output = rolling(H_output, shift)

    # print(f"{gps_index = }, {template_max_index = }, {template_max_time = }")
    # print(f"{template_start_idx = }, {template_end_idx = }, {tmp_end_idx = }")
    # print(f"{len(time_array) = }, {template_length = }, {len(H) = }")
    return {'plus': H_output.real, 'cross': -H_output.imag}



def psi4_hphc_from_generic_nojunk_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):
    """
    Generate Psi4 waveform from the given mode arrays (within waveform_arguments).

    The architecture of this model is to generate a waveform segment from the given
    NR mode arrays, and then pads with zeroes before and after it to match its length
    with the given time array.

    Important arguments in the `waveform_arguments`:
    1. "mode_array" -- list of modes (l, m) to be used in the waveform.
                        Note that in case of (l, -m) not listed, it assumes
                        the aligned spin case, and the (l, -m) mode is simply the
                        complex conjugate of the (l, m) mode (and the sign depends
                        on the oddness of l).
                        Default: PROCA_MODES = [(2,2), (2,0), (3,2), (3,3)].
    2. "hlm" -- They should be the mode arrays coming from NR simulations.
                The expected shape of them is (length, 3), i.e.: 3 long columns;
                in the format of (time, h+, h×). Examples of the "keys" can be like
                "h22" -> (2,2), "h3-2" -> (3,-2), in general, simply "h{l}{m}".
                Note that `mode_array` dominates the final modes present,
                regardless of the number of provided hlm.
                Also, "h22" must be present to define `geocent_time`.
    3. "gt0" -- If true, disregard the input `geocent_time` and set it to `start_time`
                of the `time_array`. This is to align with the convension in Bilby,
                time shift is only handled during injection.
                Default: True.
    4. "max_post_merger" -- in sec. The maximal post merger duration, default is None.
                            If None, shorter of:
                                - remainging time after `geocent_time`, and
                                - tail of waveform segment
                            otherwise, shorter of:
                                - maximal post merger duration, and
                                - tail of waveform segment


    ** Important notes **
    See the important notes of `psi4_hphc_from_generic_mode_arr`, and on top of that, 
    4. The crucial feature of this function is that it retain all the waveform 
        before the merger, so no more "pre_merger_time" in this model. 
        For this reason, it is important for the user to check the mode arrays
        to be FREE of junk radiation at the beginning.
        That said, for implementation purpose, there is still a 'pre_merger_time',
        it is defined as the shorter of:
                - duration before merger in the provided Psi4 time array (scaled), and
                - time between start of `time_array` and `geocent_time`
    """

    # print(f"{time_array[0] = }, {time_array[-1] = }, {geocent_time = }")
    delta_t = time_array[1] - time_array[0]
    max_post_merger = waveform_arguments.get('max_post_merger', None)
    ## NO pre_merger_length in this model

    # Rescale the time array to physical unit, the resample it to given delta_t
    h22_time = total_mass * 4.92e-6 * waveform_arguments["h22"].T[0]
    resampled_wfm_time = np.arange(h22_time[0], h22_time[-1], delta_t)   # in sec 
    H = np.zeros(len(resampled_wfm_time), dtype=complex)      # initialise waveform

    MD_ratio = scaling / (total_mass * luminosity_distance)
    # It is mass * distance, the original formula was:
    # 100./(total_mass*4.9254184936691445e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)
    # One can check with def. of `scaling` and simplify the above expression

    # The main reason to include this mode_array is
    # because I don't want to decode the dictionary keys to get (l, m)
    # It also has the benefits of controlling the modes without messing with the mode dictionary
    mode_array = waveform_arguments.get('mode_array', PROCA_MODES)
    for mode in mode_array:
        # Looping over each given modes
        l, m = mode
        key = f'h{l}{m}'
        h_mode = waveform_arguments.get(key, None)

        if h_mode is None:
            continue

        scaled_wfm_time = total_mass * 4.92e-6 * h_mode.T[0]
        resample_h_mode = fromNRtoPhysicalUnits(
            h_mode, scaled_wfm_time, resampled_wfm_time, MD_ratio)
        H += Ydirect(theta_jn, phase, l, m) * resample_h_mode

        if ( m != 0 ) and ( (l, -m) not in mode_array ):
            resample_h_mode_conj = (-1)**l * np.conjugate(resample_h_mode)
            H += Ydirect(theta_jn, phase, l, -m) * resample_h_mode_conj
        if tuple(mode) == (2, 2):
            template_max_index = np.argmax(np.abs(resample_h_mode))

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    # Check if the given time array contains template start time
    is_contained = gps_index > template_max_index
    template_start_idx = 0 if is_contained else \
                       template_max_index - gps_index

    # Determine length of waveform segment to retain
    if max_post_merger is not None:
        # tmp_end_idx = np.argmin(np.abs(resampled_wfm_time - (template_max_time + max_post_merger)))
        tmp_end_idx = template_max_index + round(max_post_merger / delta_t)
        # plus 1 index to match when max_post_merger is longer than the waveform tail
        template_end_idx = min(tmp_end_idx + 1, len(H))
    else:
        # Take the shorter of the two:
        # Time_array tail length: len(time_array) - gps_index
        # Waveform tail length:   len(H) - template_max_index
        template_end_idx = min(len(time_array) - gps_index + template_max_index, len(H))
    template_length = template_end_idx - template_start_idx
    # peak_to_start_len = template_max_index - template_start_idx

    # Initialise the final output waveform
    ta_len = len(time_array)
    H_output = np.zeros(ta_len, dtype=complex)
    # Place the waveform segment at the beginning of the output array
    H_output[:template_length] = H[template_start_idx : template_end_idx]
    # Apply tukey window to 1/3 of pre merger
    duration = time_array[-1] - time_array[0]
    alpha = (template_max_index - template_start_idx) * delta_t / duration * (2/3)
    # (max - start) * delta_t / 2 / (time_array[-1] / 2)
    window = tukey(ta_len, alpha=alpha)
    H_output = H_output * window

    # (temp_max_index - temp_start_idx) is the new peak idx after trimming the starting segment
    gt0 = waveform_arguments.get('gt0', True)
    gps_idx = 0 if gt0 else gps_index
    shift =  gps_idx - (template_max_index - template_start_idx)
    # Shift the waveform to the appropriate position
    if shift != 0:
        H_output = rolling(H_output, shift)

    # print(f"{gps_index = }, {template_max_index = }, {template_max_time = }")
    # print(f"{template_start_idx = }, {template_end_idx = }, {tmp_end_idx = }")
    # print(f"{len(time_array) = }, {template_length = }, {len(H) = }")
    return {'plus': H_output.real, 'cross': -H_output.imag}


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

    h22s = deprecated_fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = deprecated_fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = deprecated_fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h33 is not None:
        h33s = deprecated_fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
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

    original_re_fs = original_re_ts.to_frequencyseries()
    original_im_fs = original_im_ts.to_frequencyseries()

    corrected_re = derivative_to_differencing_FD(
            original_re_fs.data, original_re_fs.sample_frequencies, delta_t)
    corrected_im = derivative_to_differencing_FD(
            original_im_fs.data, original_im_fs.sample_frequencies, delta_t)

    dF = original_re_fs.delta_f

    corrected_re_fs = FrequencySeries(corrected_re, dF)
    corrected_im_fs = FrequencySeries(corrected_im, dF)

    corrected_re_ts = corrected_re_fs.to_timeseries().data
    corrected_im_ts = corrected_im_fs.to_timeseries().data

    return {'plus': corrected_re_ts, 'cross': corrected_im_ts}


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

    luminosity_distance_SI = luminosity_distance * 1e6 * PC_SI
    mass_1 = mass_1 * MSUN_SI
    mass_2 = mass_2 * MSUN_SI

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
    redshift_term = (1./C_SI)*1000*H0*luminosity_distance*distance_drift
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

