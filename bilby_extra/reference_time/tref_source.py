import numpy as np

from bilby.gw import utils
from bilby.gw.utils import lalsim_GetApproximantFromString
from bilby.gw.source import _base_lal_cbc_fd_waveform

from lal import MTSUN_SI
from lalsimulation import \
    SimInspiralPrecessingNRSur_get_omega_ref, \
    SimInspiralTransformPrecessingNewInitialConditions

from ..utils.logging import setup_logger
logger = setup_logger()

def get_fref_from_tref(reference_time:float, mass_1:float, mass_2:float, theta_jn:float,
                       phase:float, a_1:float, a_2:float, tilt_1:float, tilt_2:float,
                       phi_12:float, phi_jl:float, approximant='NRSur7dq4',
                       iterations:int=1, return_spins:bool=False):
    """ Compute the reference frequency from a reference (geometric) time

    One can verify the result from this function by feeding the output f_ref to
        lalsimulation.SimInspiralPrecessingNRSur_get_t_ref
    with the returned spins, and compare the t_ref with the desired value.

    Parameters
    ==========
    reference_time: The time (in geometric unit) we wish the reference frame to be.
    mass_1:         The mass of the heavier object in solar masses
    mass_2:         The mass of the lighter object in solar masses
    a_1:            Dimensionless primary spin magnitude
    tilt_1:         Primary tilt angle
    phi_12:         Azimuthal angle between the component spins
    a_2:            Dimensionless secondary spin magnitude
    tilt_2:         Secondary tilt angle
    phi_jl:         Azimuthal angle between the total and orbital angular momenta
    theta_jn:       Orbital inclination
    phase:          The phase at coalescence
    approximant:    The approximant of which the surrogate data to use,
                    either NRSur7dq4 or one of NRSurrogate family
    iterations:     Number of iterations to undergo before returning f_ref.
                    Typically, one iteration can already provide sufficient convergence.
    return_spins:   Whether to return the spin components from the last iteration.
    """


    if approximant != 'NRSur7dq4':
        approx = lalsim_GetApproximantFromString(approximant)
    else:
        approx = 93 ## NRSur7dq4

    Mtot = mass_1 + mass_2
    Q = mass_1 / mass_2

    # First construct a guess spin components
    a1t = a_1 * np.sin(tilt_1)
    a2t = a_2 * np.sin(tilt_2)
    s1x = a1t * np.cos(phase)
    s1y = a1t * np.sin(phase)
    s1z = a_1 * np.cos(tilt_1)
    s2x = a2t * np.cos(phi_12 + phase)
    s2y = a2t * np.sin(phi_12 + phase)
    s2z = a_2 * np.cos(tilt_1)

    # Then get a rough approximate of reference frequency
    omega = SimInspiralPrecessingNRSur_get_omega_ref(reference_time, Q,
                            s1x, s1y, s1z, s2x, s2y, s2z, 1,0,0,0, 0, approx)
    f_ref = omega / np.pi / (Mtot * MTSUN_SI)

    # Iterate to get better spin components and hence reference frequency
    for _ in range(iterations):
        _, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, f_ref, phase
        )
        omega = SimInspiralPrecessingNRSur_get_omega_ref(reference_time,
                        Q, s1x, s1y, s1z, s2x, s2y, s2z, 1,0,0,0, 0, approx)
        f_ref = omega / np.pi / (Mtot * MTSUN_SI)

    if return_spins:
        return f_ref, (s1x, s1y, s1z, s2x, s2y, s2z)

    return f_ref

def lal_NRSur_tref_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation
    
    ** This model does not accept reference_frequency as argument **
    ** It is also intended to serve the NRSurrogate approximants only **

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_time
          In geometric unit, and is expected to be negative. Default: -100.
          It will be converted to the f_ref according to the input parameters,
          then feed into LALInspiral via the reference_frequency option.
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.
        - lal_waveform_dictionary:
          A dictionary (lal.Dict) of arguments passed to the lalsimulation
          waveform generator. The arguments are specific to the waveform used.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='NRSur7dq4', reference_time=-100, 
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)

    input_fref = kwargs.pop('reference_frequency', None)
    if input_fref is not None:
        err_msg = 'This model does not accept reference_frequency as input, removing it.'
        pass

    waveform_kwargs.update(kwargs)
    if waveform_kwargs['waveform_approximant'] not in ['NRSur4dq2', 'NRSur7dq4']:
        err_msg = 'The reference frequency conversion function is not ' + \
                'for approximant outside the NRSurrogate family.'
        raise ValueError(err_msg)
        
    t_ref = waveform_kwargs.pop('reference_time')
    approx = waveform_kwargs['waveform_approximant']
            
    f_ref = get_fref_from_tref(t_ref, mass_1, mass_2, theta_jn, phase, 
                               a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                               approximant=approx)
    
    waveform_kwargs['reference_frequency'] = f_ref
    
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
