'''
This is intended to replace `EdGB_waveform.py` once all the dependent PE runs have completed.
As of now, besides EdGB, there is also waveform for the,
(Extended) Scalar-Tensor-Gauss-Bonnet theory,
implemented only the phase correction, up to -1 PN order.
Hopefully we will include more ST-theories waveforms.

Samson Leong -- 27th Nov, 22

12th Dec, 22: Rename cbc -> IMR, constrast with Inspiral
'''

import numpy as np
import lal
import lalsimulation as lalsim

from lal import \
    SpinWeightedSphericalHarmonic, \
    ResizeCOMPLEX16TimeSeries, \
    MTSUN_SI, MRSUN_SI, MSUN_SI, PC_SI
from lalsimulation import \
    SimIMRPhenomXPHMModes, \
    GetApproximantFromString, \
    SimInspiralChooseTDModes

from bilby.gw.source import _base_lal_cbc_fd_waveform
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from numpy.fft import fft as npfft
from numpy.fft import fftfreq

# SL: Refactoring the waveform to conform with Bilby style.
def EdGB_IMR_waveform(
        frequency_array, sqrt_alpha_bar_EdGB, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)

    waveform_kwargs.update(kwargs)
    PNa_ppE = waveform_kwargs.pop('PNa_ppE', -2)
    PNb_ppE = waveform_kwargs.pop('PNb_ppE', -7)
    # SL: These two options control corrections to phase and amplitude
    amp_corr = waveform_kwargs.pop('amp_corr', True)
    phase_corr = waveform_kwargs.pop('phase_corr', True)

    total_mass = mass_1 + mass_2
    if waveform_kwargs['waveform_approximant'] == 'TaylorF2':
        pi_M = np.pi * total_mass * MTSUN_SI
        f_ISCO = 1/6**(3/2) / pi_M
        if waveform_kwargs['maximum_frequency'] > f_ISCO:
            waveform_kwargs['maximum_frequency'] = f_ISCO

    GR_hpc_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

    total_mass2 = total_mass * total_mass
    total_mass4 = total_mass2 * total_mass2
    eta = mass_1 * mass_2 / total_mass2
    chirp_mass_T = total_mass * eta**(3/5) * MTSUN_SI

    # SL: We expect the sqrt. α in unit of km, convert it to metre.
    sq_alpha_bar_EdGB = (sqrt_alpha_bar_EdGB * 1e3)**4
    zeta_EdGB = 16*np.pi*sq_alpha_bar_EdGB/(total_mass*MRSUN_SI)**4

    sq_a_1 = a_1 * a_1
    sq_a_2 = a_2 * a_2
    s1_EdGB = 2*(np.sqrt(1 - sq_a_1) + sq_a_1 - 1)/sq_a_1
    s2_EdGB = 2*(np.sqrt(1 - sq_a_2) + sq_a_2 - 1)/sq_a_2

    # SL: Defining this constant which looks somewhat like effective spin to me
    #     feel free to changing if you see fit.
    EdGB_eff_spin = -5 * zeta_EdGB * (mass_1**2*s2_EdGB-mass_2**2*s1_EdGB)**2/(total_mass4*eta**(18/5))
    alpha_EdGB = EdGB_eff_spin / 192 if amp_corr else 0
    beta_EdGB = EdGB_eff_spin / 7168 if phase_corr else 0

    u = (np.pi * chirp_mass_T * frequency_array)**(1/3)
    length = len(u)
    ua_ppE, ub_ppE = np.zeros((2, length), dtype=float)
    ua_ppE[1:] = u[1:]**PNa_ppE
    ub_ppE[1:] = u[1:]**PNb_ppE
    delta_Psi = beta_EdGB * ub_ppE
    _EdGB_correction = (1 + alpha_EdGB * ua_ppE)*np.exp(1j*delta_Psi)

    # SL: Apply correction only to the inspiral part (up to ISCO).
    piM = np.pi * total_mass * MTSUN_SI
    f_ISCO = 6**(-3/2) / piM
    freq_mask = frequency_array < f_ISCO
    f_ISCO_idx = np.sum(freq_mask)
    EdGB_correction = np.ones(length, dtype=complex)
    EdGB_correction[:f_ISCO_idx] = _EdGB_correction[:f_ISCO_idx]

    EdGB_hp = GR_hpc_dict['plus'] * EdGB_correction
    EdGB_hc = GR_hpc_dict['cross'] * EdGB_correction
    return {'plus': EdGB_hp, 'cross': EdGB_hc}


# SL: Only consider the inspiral portion
def EdGB_inspiral_waveform(
        frequency_array, sqrt_alpha_bar_EdGB, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)

    waveform_kwargs.update(kwargs)
    PNa_ppE = waveform_kwargs.pop('PNa_ppE', -2)
    PNb_ppE = waveform_kwargs.pop('PNb_ppE', -7)
    # SL: These two options control corrections to phase and amplitude
    amp_corr = waveform_kwargs.pop('amp_corr', True)
    phase_corr = waveform_kwargs.pop('phase_corr', True)
    f_lowpass = waveform_kwargs.pop('f_lowpass', None)

    total_mass = mass_1 + mass_2
    if waveform_kwargs['waveform_approximant'] == 'TaylorF2':
        pi_M = np.pi * total_mass * MTSUN_SI
        f_ISCO = 1/6**(3/2) / pi_M
        if waveform_kwargs['maximum_frequency'] > f_ISCO:
            waveform_kwargs['maximum_frequency'] = f_ISCO

    GR_hpc_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

    total_mass2 = total_mass * total_mass
    total_mass4 = total_mass2 * total_mass2
    eta = mass_1 * mass_2 / total_mass2
    chirp_mass_T = total_mass * eta**(3/5) * MTSUN_SI

    # SL: We expect the sqrt. α in unit of km, convert it to metre.
    sq_alpha_bar_EdGB = (sqrt_alpha_bar_EdGB * 1e3)**4
    zeta_EdGB = 16*np.pi*sq_alpha_bar_EdGB/(total_mass*MRSUN_SI)**4

    sq_a_1 = a_1 * a_1
    sq_a_2 = a_2 * a_2
    s1_EdGB = 2*(np.sqrt(1 - sq_a_1) + sq_a_1 - 1)/sq_a_1
    s2_EdGB = 2*(np.sqrt(1 - sq_a_2) + sq_a_2 - 1)/sq_a_2

    # SL: Defining this constant which looks somewhat like effective spin to me
    #     feel free to changing if you see fit.
    EdGB_eff_spin = -5 * zeta_EdGB * (mass_1**2*s2_EdGB-mass_2**2*s1_EdGB)**2/(total_mass4*eta**(18/5))
    alpha_EdGB = EdGB_eff_spin / 192 if amp_corr else 0
    beta_EdGB = EdGB_eff_spin / 7168 if phase_corr else 0

    u = (np.pi * chirp_mass_T * frequency_array)**(1/3)
    length = len(u)
    ua_ppE, ub_ppE = np.zeros((2, length), dtype=float)
    ua_ppE[1:] = u[1:]**PNa_ppE
    ub_ppE[1:] = u[1:]**PNb_ppE
    delta_Psi = beta_EdGB * ub_ppE
    _EdGB_correction = (1 + alpha_EdGB * ua_ppE)*np.exp(1j*delta_Psi)

    # SL: Apply correction only to the inspiral part (up to ISCO).
    piM = np.pi * total_mass * MTSUN_SI
    f_ISCO = 6**(-3/2) / piM
    freq_mask = frequency_array < f_ISCO
    f_ISCO_idx = np.sum(freq_mask)
    EdGB_correction = np.ones(length, dtype=complex)
    EdGB_correction[:f_ISCO_idx] = _EdGB_correction[:f_ISCO_idx]

    if f_lowpass is not None:
        freq_mask = frequency_array < f_lowpass

    # SL: Key difference is the application of the freq_mask
    EdGB_hp = GR_hpc_dict['plus'] * EdGB_correction * freq_mask
    EdGB_hc = GR_hpc_dict['cross'] * EdGB_correction * freq_mask
    return {'plus': EdGB_hp, 'cross': EdGB_hc}


# Turns out Yagi considered s2 = 0.0 for NS
# Also reformat internals for speed up
def EdGB_NSBH_inspiral_waveform(
        frequency_array, sqrt_alpha_bar_EdGB, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)

    waveform_kwargs.update(kwargs)
    PNa_ppE = waveform_kwargs.pop('PNa_ppE', -2)
    PNb_ppE = waveform_kwargs.pop('PNb_ppE', -7)
    # SL: These two options control corrections to phase and amplitude
    amp_corr = waveform_kwargs.pop('amp_corr', True)
    phase_corr = waveform_kwargs.pop('phase_corr', True)
    f_lowpass = waveform_kwargs.pop('f_lowpass', None)

    total_mass = mass_1 + mass_2
    piM = np.pi * total_mass * MTSUN_SI
    f_ISCO = 6**(-3/2) / piM

    if (waveform_kwargs['waveform_approximant'] == 'TaylorF2') and \
            (waveform_kwargs['maximum_frequency'] > f_ISCO):
        waveform_kwargs['maximum_frequency'] = f_ISCO

    GR_hpc_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

    total_mass2 = total_mass * total_mass
    total_mass4 = total_mass2 * total_mass2
    eta = mass_1 * mass_2 / total_mass2

    # SL: We expect the sqrt. α in unit of km, convert it to metre.
    coupling = -5 * np.pi * (sqrt_alpha_bar_EdGB * 1e3 / (total_mass*MRSUN_SI))**4

    sq_a_1 = a_1 * a_1
    sq_a_2 = a_2 * a_2
    s1_EdGB = 2*(np.sqrt(1 - sq_a_1) + sq_a_1 - 1)/sq_a_1
    # s2_EdGB = 2*(np.sqrt(1 - sq_a_2) + sq_a_2 - 1)/sq_a_2
    s2_EdGB = 0.0

    # SL: Defining this constant which looks somewhat like effective spin to me
    #     feel free to changing if you see fit.
    EdGB_eff_spin = coupling * (mass_1**2*s2_EdGB-mass_2**2*s1_EdGB)**2/(total_mass4*eta**5)
    alpha_EdGB = EdGB_eff_spin / 12 if amp_corr else 0      # 192 = 16 * 12
    beta_EdGB = EdGB_eff_spin / 448 if phase_corr else 0    # 7168 = 16 * 448

    u = (np.pi * total_mass * MTSUN_SI * frequency_array)**(1/3)
    length = len(u)
    ua_ppE, ub_ppE = np.zeros((2, length), dtype=float)
    ua_ppE[1:] = u[1:]**PNa_ppE
    ub_ppE[1:] = u[1:]**PNb_ppE
    delta_Psi = beta_EdGB * ub_ppE
    _EdGB_correction = (1 + alpha_EdGB * ua_ppE)*np.exp(1j*delta_Psi)

    # SL: Apply correction only to the inspiral part (up to ISCO).
    freq_mask = frequency_array < f_ISCO
    f_ISCO_idx = np.sum(freq_mask)
    EdGB_correction = np.ones(length, dtype=complex)
    EdGB_correction[:f_ISCO_idx] = _EdGB_correction[:f_ISCO_idx]

    if f_lowpass is not None:
        freq_mask = frequency_array < f_lowpass

    # SL: Key difference is the application of the freq_mask
    EdGB_hp = GR_hpc_dict['plus'] * EdGB_correction * freq_mask
    EdGB_hc = GR_hpc_dict['cross'] * EdGB_correction * freq_mask
    return {'plus': EdGB_hp, 'cross': EdGB_hc}


def EdGB_NSBH_highPN_inspiral_waveform(
        frequency_array, sqrt_alpha_bar_EdGB, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)

    waveform_kwargs.update(kwargs)
    # SL: we use all available PN orders here, skipping
    # PNa_ppE = waveform_kwargs.pop('PNa_ppE', -2)
    # PNb_ppE = waveform_kwargs.pop('PNb_ppE', -7)
    # SL: These two options control corrections to phase and amplitude
    # SL: amplitude is not available yet, disregarding input
    # amp_corr = waveform_kwargs.pop('amp_corr', True)
    amp_corr = False
    phase_corr = waveform_kwargs.pop('phase_corr', True)
    f_lowpass = waveform_kwargs.pop('f_lowpass', None)

    total_mass = mass_1 + mass_2
    piM = np.pi * total_mass * MTSUN_SI
    f_ISCO = 6**(-3/2) / piM

    if (waveform_kwargs['waveform_approximant'] == 'TaylorF2') and \
            (waveform_kwargs['maximum_frequency'] > f_ISCO):
        waveform_kwargs['maximum_frequency'] = f_ISCO

    GR_hpc_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

    # Get higher order PN phase corrections as in:
    # https://arxiv.org/pdf/2201.02543.pdf
    # Later refer to as "the paper"

    sq_a_1 = a_1 * a_1
    sq_a_2 = a_2 * a_2
    s1 = 2*(np.sqrt(1 - sq_a_1) + sq_a_1 - 1)/sq_a_1
    # s2 = 2*(np.sqrt(1 - sq_a_2) + sq_a_2 - 1)/sq_a_2
    s2 = 0   # Sensitivity for NS
    f3, f4 = 0, 0  # Unknown tensor non-dipole emission

    total_mass2 = total_mass * total_mass
    total_mass4 = total_mass2 * total_mass2
    eta = mass_1 * mass_2 / total_mass2
    eta2 = eta * eta
    eta3 = eta2 * eta

    # SL: Defining this constant which looks somewhat like effective spin to me
    #     feel free to changing if you see fit.
    Eff_spin_m = - 5 * np.pi * (mass_1**2*s2-mass_2**2*s1)**2/(total_mass4*eta**5)
    Eff_spin_p = - 5 * np.pi * (mass_1**2*s2+mass_2**2*s1)**2/(total_mass4*eta**5)
    s1s2_eta = np.pi * (s1*s2/eta3)

    ## Comppute the correction coefficients
    c_m1  = Eff_spin_m / 448
    c_0   = Eff_spin_m / 43008 * (659 + 728*eta) - (5/16) * s1s2_eta
    c_p05 = Eff_spin_m * (-15*np.pi/448)
    c_p1  = Eff_spin_p / 48384 * (535 + 924*eta) + (5/576) * Eff_spin_m * (
                12_497_995/1_016_064 - (11/2) * (mass_1-mass_2)/total_mass * Eff_spin_p/Eff_spin_m
                + (15407/1440)*eta + (165/16)*eta2)
    c_p15 = Eff_spin_m * (-np.pi / 10) - 6 * np.pi * s1s2_eta + (3/32) * f3/eta
    # c_p2 is intentionally skipped for its small magnitude and complex expression


    ## Assemble the ppE phase
    # The actual coupling should begin with 16π, removing as in the paper
    # SL: We expect the sqrt. α in unit of km, convert it to metre.
    coupling = (sqrt_alpha_bar_EdGB * 1e3)**4 / (total_mass*MRSUN_SI)**4

    # SL: Apply correction only to the inspiral part (up to ISCO).
    freq_mask = frequency_array < f_ISCO

    length = len(frequency_array)
    # Change the expansion to use total_mass, same as in the paper
    v = (np.pi * total_mass * MTSUN_SI * frequency_array)**(1/3)
    dPhase = np.zeros(length)
    # dPhase_PN = {}
    coef_pairs = [(-1, c_m1), (0, c_0), (0.5, c_p05), (1, c_p1), (1.5, c_p15)]
    for idx, coef in coef_pairs:
        dphase_i = coupling * coef * v[1:]**(-5 + 2*idx)
        # dPhase_PN[idx] = dphase_i
        dPhase[1:] += dphase_i

    dPhase *= freq_mask

    if f_lowpass is not None:
        freq_mask = frequency_array < f_lowpass

    # SL: Key difference is the application of the freq_mask
    EdGB_hp = GR_hpc_dict['plus'] * np.exp(1j*dPhase) * freq_mask
    EdGB_hc = GR_hpc_dict['cross'] * np.exp(1j*dPhase) * freq_mask
    return {'plus': EdGB_hp, 'cross': EdGB_hc}


def XPHM_modes_dict(modes=None):
    lalDict = lal.CreateDict()
    mode_array_lal = lalsim.SimInspiralCreateModeArray()
    std_modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4),
                 (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]
    modes = std_modes if modes is None else modes
    for ell, m in modes:
        lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, ell, m)
    lalsim.SimInspiralWaveformParamsInsertModeArray(lalDict, mode_array_lal)
    return lalDict


def modes_dict(modes=None):
    lalDict = lal.CreateDict()
    mode_array_lal = lalsim.SimInspiralCreateModeArray()

    if modes is None:
        NRSur_ell_MAX = 4
        ell = 2
        while ell <= NRSur_ell_MAX:
            for m in range(-ell, ell+1):
                # print(ell, m)
                lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, ell, m)
            ell += 1
    elif type(modes) == list:
        for indiv_lm in modes:
            ell, m = indiv_lm
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, ell, m)
    else:
        ell, m = modes
        lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, ell, m)

    lalsim.SimInspiralWaveformParamsInsertModeArray(lalDict, mode_array_lal)
    return lalDict


def lal_XPHM_STGB_waveform(frequency_array: list, charge_1, charge_2, mass_1, mass_2, luminosity_distance,
                           a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs) -> dict:

    total_mass = mass_1 + mass_2
    eta = mass_1 * mass_2 / total_mass / total_mass
    azimuth = np.pi / 2 - phase

    m1_SI = mass_1 * MSUN_SI * 1.
    m2_SI = mass_2 * MSUN_SI * 1.

    delta_f = frequency_array[1] - frequency_array[0]
    f_min = kwargs.get('minimum_frequency', 10.)
    f_max = kwargs.get('maximum_frequency', frequency_array[-1])
    f_ref = kwargs.get('reference_frequency', 11.)

    cart_spins = bilby_to_lalsimulation_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                                     a_1, a_2, m1_SI, m2_SI, f_ref, phase)

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (float(para) for para in cart_spins)

    xphm_hlms = SimIMRPhenomXPHMModes(
        m1_SI = m1_SI, m2_SI = m2_SI,
        S1x = spin_1x, S1y = spin_1y, S1z = spin_1z,
        S2x = spin_2x, S2y = spin_2y, S2z = spin_2z,
        deltaF = delta_f, f_min = f_min, f_max = f_max,
        f_ref = f_ref, phiRef = phase, distance = luminosity_distance * 1e6 * PC_SI,
        inclination = theta_jn, LALparams=XPHM_modes_dict()
    )

    # SL: Judging from the paper, these charges carry the dimension of length
    #     in the following code, we assume it has units of 'm'.
    mass_1R = mass_1 * MRSUN_SI
    mass_2R = mass_2 * MRSUN_SI
    STGB_corr_amp = 5 / 14336 / eta * (charge_1/mass_1R - charge_2/mass_2R)**2
    piM = 2 * np.pi * total_mass * MTSUN_SI

    STGB_xphm_hp = 0 + 0j
    STGB_xphm_hc = 0 + 0j
    while xphm_hlms is not None:
        ell, mm = (xphm_hlms.l, xphm_hlms.m)
        GR_xphm_hlm = xphm_hlms.mode.data.data

        total_length = len(GR_xphm_hlm)
        zero_index = int((total_length -1)/2)

        GR_hlm_pos = GR_xphm_hlm[zero_index:]
        GR_hlm_neg = np.flip(GR_xphm_hlm[:zero_index + 1])

        # SL: Jesus, I can't believe they actually return with positive and negative frequencies....
        if mm != 0:
            freq_arr = xphm_hlms.fdata.data
            freqs_pos = freq_arr[zero_index:]

            f_ISCO = 6 ** (-3/2) / piM
            ISCO_idx = np.argmin(np.abs(freq_arr - f_ISCO))

            rel_sep = (piM * freqs_pos[1:ISCO_idx] / np.abs(mm)) ** (-7/3)

            STGB_corr = np.ones(zero_index+1, dtype=complex)
            STGB_corr[1:ISCO_idx] = np.exp(-1j * mm * STGB_corr_amp * rel_sep)

            # SL: Not entire sure if this should be theta_jn or iota
            # SL: Yes, it is theta_jn, same as the XPHM code
            Ylm = SpinWeightedSphericalHarmonic(theta_jn, azimuth, -2, ell, mm)
            cYlm = np.conjugate(Ylm)

        else:
            STGB_corr = np.ones(zero_index+1, dtype=complex)

        STGB_xphm_hp += (Ylm * GR_hlm_pos + cYlm * np.conjugate(GR_hlm_neg)) * STGB_corr / 2
        STGB_xphm_hc += (Ylm * GR_hlm_pos - cYlm * np.conjugate(GR_hlm_neg)) * STGB_corr / 2 * 1j

        xphm_hlms = xphm_hlms.next

    return {'plus': STGB_xphm_hp/2, 'cross': 1j*STGB_xphm_hc/2}


def lal_NRSur_STGB_FD_waveform(frequency_array: list, charge_1, charge_2, mass_1, mass_2, luminosity_distance,
                           a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs) -> dict:

    total_mass = mass_1 + mass_2
    eta = mass_1 * mass_2 / total_mass / total_mass
    azimuth = np.pi / 2 - phase

    m1_SI = mass_1 * MSUN_SI * 1.
    m2_SI = mass_2 * MSUN_SI * 1.

    delta_f = frequency_array[1] - frequency_array[0]
    f_min = kwargs.get('minimum_frequency', 10.)
    f_max = kwargs.get('maximum_frequency', frequency_array[-1])
    f_ref = kwargs.get('reference_frequency', 11.)
    srate = kwargs.get('sampling_frequency', 1024.)
    lalparams = kwargs.get('LALparams', modes_dict())
    waveform_approximant = kwargs.get('waveform_approximant', 'NRSur7dq4')

    approximant = GetApproximantFromString(waveform_approximant)
    chirplen = 2 * f_max / delta_f

    cart_spins = bilby_to_lalsimulation_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                                     a_1, a_2, m1_SI, m2_SI, f_ref, phase)

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (float(para) for para in cart_spins)

    TD_hlms = SimInspiralChooseTDModes(
        phiRef=0.0, deltaT=1/srate,
        m1 = m1_SI, m2 = m2_SI,
        S1x = spin_1x, S1y = spin_1y, S1z = spin_1z,
        S2x = spin_2x, S2y = spin_2y, S2z = spin_2z,
        f_min = f_min, f_ref = f_ref,
        r = luminosity_distance * 1e6 * PC_SI,
        lmax = 4, LALpars=lalparams, approximant=approximant
    )

    # SL: Judging from the paper, these charges carry the dimension of length
    #     in the following code, we assume it has geometric length unit.
    STGB_corr_amp = 5 / 14336 / eta * (charge_1/mass_1 - charge_2/mass_2)**2
    piM = 2 * np.pi * total_mass * MTSUN_SI

    STGB_nrsur_hp = 0 + 0j
    STGB_nrsur_hc = 0 + 0j

    while TD_hlms is not None:
        ell, mm = (TD_hlms.l, TD_hlms.m)
        # This resize is done to the modes directly
        ResizeCOMPLEX16TimeSeries(TD_hlms.mode, int(TD_hlms.mode.data.length - chirplen), int(chirplen))

        GR_TD_hlm = TD_hlms.mode.data.data

        if (ell, mm) == (2, 2):
            epoch_time = TD_hlms.mode.epoch
       #  print(TD_hlms_padded.sampleUnits, TD_hlms_padded.data.length)

        N = TD_hlms.mode.data.length
        half_N = int(N/2)

        GR_FD_hlm = npfft(GR_TD_hlm) / srate
        freq_array = fftfreq(N, d=TD_hlms.mode.deltaT)
        # Unpacking numpy standards, see implementation detail:
        # https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details
        GR_FD_hlm_pos = np.append(GR_FD_hlm[0], GR_FD_hlm[1:half_N+1])
        GR_FD_hlm_neg = np.append(GR_FD_hlm[0], np.flip(GR_FD_hlm[half_N:]))
        # Since freq[N/2] correspond to both +ve and -ve Nyquist freq.,
        # it is necessary to regularise it
        # f = 0 Hz is excluded
        freq_pos = np.append(freq_array[1:half_N], np.abs(freq_array[half_N]))

        # SL: Jesus, I can't believe they actually return with positive and negative frequencies....
        if mm != 0:

            f_ISCO = 6 ** (-3/2) / piM
            ISCO_idx = np.argmin(np.abs(freq_pos - f_ISCO))

            rel_sep = (piM * freq_pos[:ISCO_idx] / np.abs(mm)) ** (-7/3)

            STGB_corr = np.ones(half_N+1, dtype=complex)
            STGB_corr[1:ISCO_idx+1] = np.exp(-1j * mm * STGB_corr_amp * rel_sep)

            # SL: Not entire sure if this should be theta_jn or iota
            # SL: Yes, it is theta_jn, same as the XPHM code
            Ylm = SpinWeightedSphericalHarmonic(iota, azimuth, -2, ell, mm)
            cYlm = np.conjugate(Ylm)

        else:
            STGB_corr = np.ones(half_N+1, dtype=complex)

        STGB_nrsur_hp += (Ylm * GR_FD_hlm_pos + cYlm * np.conjugate(GR_FD_hlm_neg)) * STGB_corr
        STGB_nrsur_hc += (Ylm * GR_FD_hlm_pos - cYlm * np.conjugate(GR_FD_hlm_neg)) * STGB_corr

        TD_hlms = TD_hlms.next

    # SL: Bilby has this final time-shift for TD to FD models, not sure why
    # tho. But it is needed to align the merger time.
    dt = 1 / delta_f + (epoch_time.gpsSeconds + epoch_time.gpsNanoSeconds * 1e-9)
    freq_pos0 = np.append(0, freq_pos)
    time_shift = np.exp(-1j * 2 * np.pi * dt * freq_pos0)

    freq_bound = (f_min <= freq_pos0) * (freq_pos0 <= f_max)

    STGB_nrsur_hp *= time_shift * freq_bound / 2
    STGB_nrsur_hc *= time_shift * freq_bound / 2 * 1j

    return {'plus': STGB_nrsur_hp, 'cross': STGB_nrsur_hc}
