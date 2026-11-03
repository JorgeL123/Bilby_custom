import numpy as np
from typing import Type
from numbers import Number
from bilby.gw.result import CBCResult
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions, \
                            SimInspiralTransformPrecessingWvf2PE
from lal import MSUN_SI

import logging
logger = logging.getLogger('Bilby-Extra')

vectorised_SimInspiralTransformPrecessingNewInitialConditions = np.vectorize(
    SimInspiralTransformPrecessingNewInitialConditions
)
vectorised_SimInspiralTransformPrecessingWvf2PE = np.vectorize(
    SimInspiralTransformPrecessingWvf2PE
)

MASSES = ('mass_1', 'mass_2', 'total_mass', 'mass_ratio', 'chirp_mass', 'symmetric_mass_ratio')
SPIN_ANG = ('a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl')
SPIN_COM = ('spin_1x', 'spin_2x', 'spin_1y', 'spin_2y', 'spin_1z', 'spin_2z')
ORIENTATION = ('phase', 'iota', 'theta_jn')
INTRINSIC = MASSES + SPIN_ANG + SPIN_COM + ORIENTATION

def spin_angles_to_components(
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
    reference_frequency, phase
) -> tuple:
    if (a_1 == 0 or tilt_1 in [0, np.pi]) and (a_2 == 0 or tilt_2 in [0, np.pi]):
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1 * np.cos(tilt_1)
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2 * np.cos(tilt_2)
        iota = theta_jn
        return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z
    else:
        args = (
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
            mass_2, reference_frequency, phase
        )
        # print(args)
        float_inputs = all([isinstance(arg, Number) for arg in args])
        if float_inputs:
            func = SimInspiralTransformPrecessingNewInitialConditions
        else:
            func = vectorised_SimInspiralTransformPrecessingNewInitialConditions
        return func(*args)


def spin_components_to_angles(
    iota, phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
    spin_2x, spin_2y, spin_2z, reference_frequency
) -> tuple:
    if (spin_1x == spin_1y == 0) and (spin_2x == spin_2y == 0):
        a_1 = spin_1z
        a_2 = spin_2z
        tilt_1 = 0
        tilt_2 = 0
        phi_jl = 0
        phi_12 = 0
        theta_jn = iota
        return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2
    else:
        if mass_1 + mass_2 > MSUN_SI:
            logger.warning("Input masses are not in solar masses, assuming them in SI_units.")
            mass_1_SM = mass_1 / MSUN_SI
            mass_2_SM = mass_2 / MSUN_SI
        else:
            mass_1_SM = mass_1
            mass_2_SM = mass_2
        # Yep, they chose to abide to solar masses in this case.
        args = (
            iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
            mass_1_SM, mass_2_SM, reference_frequency, phase
        )
        float_inputs = all([isinstance(arg, Number) for arg in args])
        if float_inputs:
            func = SimInspiralTransformPrecessingWvf2PE
        else:
            func = vectorised_SimInspiralTransformPrecessingWvf2PE
        return func(*args)


class ParamProperty(object):
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, _value):
        if _value is None:
            return
        value = float(_value) \
                if isinstance(_value, Number) \
                else np.array(_value)
        setattr(obj, self.private_name, value)


class BBHIntrinsic(object):
    """
    To be honest, I am not familiar with Python classes, and I
    don't fully understand what I am doing here. Especially
    those before the __init__ function
    Reference: https://stackoverflow.com/a/71751702
    """

    for key in INTRINSIC:
        locals()[key] = ParamProperty()

    def __init__(self, parameters: dict, fref: float = None):
        self.input_dict = parameters
        if set(MASSES).isdisjoint(set(parameters.keys())):
            logger.error('Provided parameters contain nether component masses nor mass ratio.')

        # print(self.__class__)
        # for key, val in self.input_dict.items():
        #     if val is not None:
        #         print(f'{key}: {val:.4f}')

        self.reference_frequency = float(fref)

        for key in INTRINSIC:
            _value = parameters.get(key, None)
            if _value is None:
                value = _value
            elif isinstance(_value, Number):
                value = float(_value)
            else:
                value = np.array(_value)
            setattr(self, '_'+key, value)

        self._fill_masses()
        self._check_masses_and_ratio()
        self._fill_spins()


    @property
    def all_parameters(self) -> dict:
        output_dict = self.input_dict.copy()
        for key in INTRINSIC:
            output_dict[key] = getattr(self, key)
        return output_dict


    @property
    def _ones(self):
        value = list(self.input_dict.values())[0]
        if isinstance(value, Number):
            return 1.0
        else:
            return np.ones(len(value), dtype=float)


    def physically_equil_transform(self, output_class):
        transformed = output_class(
            parameters=self.all_parameters, fref=self.reference_frequency)
        transformed.mass_ratio = 1/self.mass_ratio
        transformed.total_mass = self.total_mass
        transformed.mass_1 = self.mass_2
        transformed.mass_2 = self.mass_1
        transformed.a_1 = self.a_2
        transformed.a_2 = self.a_1
        transformed.tilt_1 = self.tilt_2
        transformed.tilt_2 = self.tilt_1
        transformed.phi_12 = 2 * np.pi - self.phi_12
        transformed.phase  = np.mod(self.phase + np.pi, 2 * np.pi)
        transformed.spin_1x = -self.spin_2x
        transformed.spin_2x = -self.spin_1x
        transformed.spin_1y = -self.spin_2y
        transformed.spin_2y = -self.spin_1y
        transformed.spin_1z = self.spin_2z
        transformed.spin_2z = self.spin_1z
        transformed.phi_jl = self.phi_jl
        transformed.iota = self.iota
        transformed.theta_jn = self.theta_jn

        return transformed


    def _fill_masses(self):
        not_have_comps = {'mass_1', 'mass_2'}.issubset(self.input_dict.keys())
        not_have_ratio = 'mass_ratio' in list(self.input_dict.keys())
        not_have_chirp = {'symmetric_mass_ratio', 'chirp_mass'}.issubset(self.input_dict.keys())

        if not_have_comps:
            q = self.mass_ratio
            self._mass_1 = self.total_mass / (1+q)
            self._mass_2 = self.total_mass - self._mass_1

        if not_have_ratio:
            self._mass_ratio = self.mass_2 / self.mass_1
            self._total_mass = self.mass_1 + self.mass_2

        if not_have_chirp:
            reduced_mass = self.mass_1 * self.mass_2 / self.total_mass
            self._symmetric_mass_ratio = reduced_mass / self.total_mass
            self._chirp_mass = self._symmetric_mass_ratio ** (3/5) * self.total_mass



    def _fill_spins(self):
        not_have_comp = set(SPIN_COM).isdisjoint(self.input_dict.keys())
        not_have_angs = set(SPIN_ANG).isdisjoint(self.input_dict.keys())

        zeroes = int(self._ones * 0)

        if not_have_comp and not_have_angs:
            logger.warning('Provided parameters contain nether component spins nor spin angles, ' +\
                          'setting them to 0.')
            for key in (SPIN_COM + SPIN_ANG):
                setattr(self, '_'+key, zeroes)
            return

        if not_have_comp:
            if self.reference_frequency is None:
                logger.warning('Reference frequency not provided, cannot compute spin components, setting them to 0.')
                for key in SPIN_COM:
                    setattr(self, '_'+key, zeroes)
            else:
                self._iota, self._spin_1x, self._spin_1y, self._spin_1z, \
                self._spin_2x, self._spin_2y, self._spin_2z = \
                spin_angles_to_components(
                    self.theta_jn, self.phi_jl, self.tilt_1, self.tilt_2, self.phi_12,
                    self.a_1, self.a_2, self.mass_1, self.mass_2, self._ones * self.reference_frequency, self.phase
                )
            return

        if not_have_angs:
            if self.reference_frequency is None:
                logger.warning('Reference frequency not provided, cannot compute spin angles, setting them to 0.')
                for key in SPIN_ANG:
                    setattr(self, '_'+key, zeroes)
            else:
                self._theta_jn, self._phi_jl, self._tilt_1, self._tilt_2, self._phi_12, self._a_1, self._a_2 = \
                spin_components_to_angles(
                    self.iota, self.phase, self.mass_1, self.mass_2, self.spin_1x, self.spin_1y, self.spin_1z,
                    self.spin_2x, self.spin_2y, self.spin_2z, self._ones * self.reference_frequency
                )
            return

        def _check_masses_and_ratio(self):
            pass


class q_BBHIntrinsic(BBHIntrinsic):
    """
    The conventions adopted by LAL:
    m1 > m2, and q = m2 / m1
    """
    def __init__(self, parameters: dict, fref: float = None):
        super().__init__(parameters=parameters, fref=fref)

    def _check_masses_and_ratio(self):
        is_flipped = all(self.mass_2 >= self.mass_1)
        is_inverse = all(self.mass_ratio >= 1)

        if not (is_flipped and is_inverse):
            return

        transformed = self.physically_equil_transform(cls)
        if is_flipped and is_inverse:
            self = transformed
        elif is_flipped:
            transformed.mass_ratio = 1/transformed.mass_ratio
        else:
            pass


    def physically_equil_transform(self):
        return super().physically_equil_transform(Q_BBHIntrinsic)


class Q_BBHIntrinsic(BBHIntrinsic):
    """
    The flipped convention:
    m2 > m1, and q = m2 / m1 > 1
    """
    def __init__(self, parameters: dict, fref: float = None):
        super().__init__(parameters=parameters, fref=fref)


    def physically_equil_transform(self):
        return super().physically_equil_transform(q_BBHIntrinsic)

