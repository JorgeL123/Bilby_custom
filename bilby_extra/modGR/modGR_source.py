import numpy as np

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from bilby.gw.utils import (lalsim_GetApproximantFromString,
                    lalsim_SimInspiralFD,
                    lalsim_SimInspiralChooseFDWaveform,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda2)


def _modGR_lal_cbc_fd_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
        lambda_1=0.0, lambda_2=0.0, eccentricity=0.0,
        dchi0=0.0, dchi1=0.0, dchi2=0.0, dchi3=0.0, dchi4=0.0,
        dchi5=0.0, dchi5l=0.0, dchi6=0.0, dchi6l=0.0,
        dchi7=0.0, dbeta2=0.0, dbeta3=0.0, dalpha2=0.0,
        dalpha3=0.0, dalpha4=0.0, **waveform_kwargs):
    """ Generate a cbc waveform model using lalsimulation
    with the addition of several PN coefficients for testing GR.
    The first testingGR paper with the use the of these coefficients:
        https://arxiv.org/pdf/1602.03841.pdf
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.221101

    Parameters
    ==========
    Most parameters are the same as `_base_lal_cbc_fd_waveform`, except for the
    dephasing coefficients, details of their implementations refers to here:
        https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_waveform_params_8h.html


    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    import lal
    import lalsimulation as lalsim

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    pn_spin_order = waveform_kwargs['pn_spin_order']
    pn_tidal_order = waveform_kwargs['pn_tidal_order']
    pn_phase_order = waveform_kwargs['pn_phase_order']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )

    approximant = lalsim_GetApproximantFromString(waveform_approximant)
    if pn_amplitude_order != 0:
        start_frequency = lalsim.SimInspiralfLow2fStart(
            minimum_frequency, int(pn_amplitude_order), approximant)
    else:
        start_frequency = minimum_frequency

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
        waveform_dictionary, int(pn_spin_order))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(
        waveform_dictionary, int(pn_tidal_order))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        waveform_dictionary, int(pn_phase_order))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
        waveform_dictionary, int(pn_amplitude_order))
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi0(
        waveform_dictionary, float(dchi0))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi1(
        waveform_dictionary, float(dchi1))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi2(
        waveform_dictionary, float(dchi2))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi3(
        waveform_dictionary, float(dchi3))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi4(
        waveform_dictionary, float(dchi4))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi5(
        waveform_dictionary, float(dchi5))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi5L(
        waveform_dictionary, float(dchi5l))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi6(
        waveform_dictionary, float(dchi6))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi6L(
        waveform_dictionary, float(dchi6l))
    lalsim.SimInspiralWaveformParamsInsertNonGRDChi7(
        waveform_dictionary, float(dchi7))
    lalsim.SimInspiralWaveformParamsInsertNonGRDBeta2(
        waveform_dictionary, float(dbeta2))
    lalsim.SimInspiralWaveformParamsInsertNonGRDBeta3(
        waveform_dictionary, float(dbeta3))
    lalsim.SimInspiralWaveformParamsInsertNonGRDAlpha2(
        waveform_dictionary, float(dalpha2))
    lalsim.SimInspiralWaveformParamsInsertNonGRDAlpha3(
        waveform_dictionary, float(dalpha3))
    lalsim.SimInspiralWaveformParamsInsertNonGRDAlpha4(
        waveform_dictionary, float(dalpha4))

    for key, value in waveform_kwargs.items():
        func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
        if func is not None:
            func(waveform_dictionary, value)

    if waveform_kwargs.get('numerical_relativity_file', None) is not None:
        lalsim.SimInspiralWaveformParamsInsertNumRelData(
            waveform_dictionary, waveform_kwargs['numerical_relativity_file'])

    if ('mode_array' in waveform_kwargs) and waveform_kwargs['mode_array'] is not None:
        mode_array = waveform_kwargs['mode_array']
        mode_array_lal = lalsim.SimInspiralCreateModeArray()
        for mode in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    if lalsim.SimInspiralImplementedFDApproximants(approximant):
        wf_func = lalsim_SimInspiralChooseFDWaveform
    else:
        wf_func = lalsim_SimInspiralFD
    try:
        hplus, hcross = wf_func(
            mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, luminosity_distance, iota, phase,
            longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
            start_frequency, maximum_frequency, reference_frequency,
            waveform_dictionary, approximant)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_2y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=start_frequency)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    h_cross = np.zeros_like(frequency_array, dtype=complex)

    if len(hplus.data.data) > len(frequency_array):
        logger.debug("LALsim waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(hplus.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating lalsim array.")
        h_plus = hplus.data.data[:len(h_plus)]
        h_cross = hcross.data.data[:len(h_cross)]
    else:
        h_plus[:len(hplus.data.data)] = hplus.data.data
        h_cross[:len(hcross.data.data)] = hcross.data.data

    h_plus *= frequency_bounds
    h_cross *= frequency_bounds

    if wf_func == lalsim_SimInspiralFD:
        dt = 1 / hplus.deltaF + (hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        h_plus[frequency_bounds] *= time_shift
        h_cross[frequency_bounds] *= time_shift

    return dict(plus=h_plus, cross=h_cross)


def modGR_lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
        dchi0, dchi1, dchi2, dchi3, dchi4, dchi5, dchi5l, dchi6,
        dchi6l, dchi7, dbeta2, dbeta3, dalpha2, dalpha3, dalpha4, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation
    An extension of `lal_binary_black_hole` with extra modGR PN coefficients.

    Parameters
    ----------
    Refers to `lal_binary_black_hole` for most parameters,
    the extra dephasing coefficients should be floats.

    From `XLALSimInspiralApproximantAcceptTestGRParams`, only these are available:
        TaylorF2, TaylorF2Ecc, TaylorF2NLTides
        PhenSpinTaylor, PhenSpinTaylorRD, SpinTaylorF2
        EccentricFD, Eccentricity, EccentricTD
        SEOBNRv4_ROM
        IMRPhenomC, IMRPhenomD, 
        IMRPhenomP, IMRPhenomPv2, IMRPhenomHM
        IMRPhenomPv2_NRTidal, IMRPhenomPv2_NRTidalv2
        IMRPhenomD_NRTidal, IMRPhenomD_NRTidalv2
        IMRPhenomPv3, IMRPhenomPv3HM

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _modGR_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, dchi0=dchi0, dchi1=dchi1, dchi2=dchi2, dchi3=dchi3,
        dchi4=dchi4, dchi5=dchi5, dchi5l=dchi5l, dchi6=dchi6, dchi6l=dchi6l,
        dchi7=dchi7, dbeta2=dbeta2, dbeta3=dbeta3, dalpha2=dalpha2,
        dalpha3=dalpha3, dalpha4=dalpha4, **waveform_kwargs)
