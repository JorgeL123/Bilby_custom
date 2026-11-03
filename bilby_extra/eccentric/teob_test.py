import sys
sys.path.insert(1, './external/TEOBResumS')
import EOBRun_module
import numpy as np

print(EOBRun_module.__file__)


def modes_to_k(modes):
    """ Map (l, m) to linear index """
    return sorted([int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes])


# Generate the target waveform
M1 = 36 # Mo
M2 = 29 # Mo
Deff = 410. # Mpc
iota = np.pi/2.
k = modes_to_k([[2,2], [2,1], [3,3], [3,2], [4,4]])# Use 22 mode only
pars = {
    'M'                  : M1+M2,
    'q'                  : M1/M2,
    'Lambda1'            : 0.,
    'Lambda2'            : 0.,
    'chi1'               : 0.4,
    'chi2'               : 0.5,
    'domain'             : 0,      # TD
    'use_mode_lm'        : k,      # List of modes to use/output through EOBRunPy
    'srate_interp'       : 4096.,  # srate at which to interpolate. Default = 4096.
    'use_geometric_units': 0,      # Output quantities in geometric units. Default = 1
    'initial_frequency'  : 10.,    # in Hz if use_geometric_units = 0, else in geometric units
    'interp_uniform_grid': 1,      # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
    'distance': Deff,
    'inclination':iota,
}
T, Hp, Hc = EOBRun_module.EOBRunPy(pars)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
plt_args = {'alpha': 0.55, 'lw': 0.9}
axes[0].plot(T, Hp, label="ecc = 0.0", **plt_args)
axes[1].plot(T, Hc, label="ecc = 0.0", **plt_args)

pars['ecc'] = 0.1
pars['ecc_freq'] = 10
T, Hp, Hc = EOBRun_module.EOBRunPy(pars)
axes[0].plot(T, Hp, label="ecc = 0.1", **plt_args)
axes[1].plot(T, Hc, label="ecc = 0.1", **plt_args)

pars['ecc'] = 0.2
pars['ecc_freq'] = 10
T, Hp, Hc = EOBRun_module.EOBRunPy(pars)
axes[0].plot(T, Hp, label="ecc = 0.2", **plt_args)
axes[1].plot(T, Hc, label="ecc = 0.2", **plt_args)


axes[0].set_title("h plus")
axes[1].set_title("h cross")
axes[0].legend()
fig.savefig("TEOB_first_test_HM.png")
