import teob_source
import numpy as np

param = {
    'mass_1': 36,
    'mass_2': 29,
    'iota':  2.1,
    'luminosity_distance': 1000,
    'spin_1z': 0.1,
    'spin_2z': 0.3,
    'phase': 0.4,
    'geocent_time': 3.8,
}

sampling_freq = 4096

ta = np.arange(0, 6, 1/sampling_freq)


higher_mode_arr = [
    (2,2), (2,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
wfm_arg = {'delta_t': 1/sampling_freq, 'f_low': 10, 'ecc_freq': 1, "mode_array": higher_mode_arr}

import matplotlib.pyplot as plt

# fig, axes = plt.subplot_mosaic([['hp', 'hc'],['abs', 'abs']], figsize=(20, 10))
# plt_args = {'alpha': 0.55, 'lw': 0.9}
# for ecc in [0.0, 0.1, 0.2]:
#     param['eccentricity'] = ecc
#     hp, hc = teob_source.teobresums_td(ta, **param, **wfm_arg)
#     axes['hp'].plot(ta, hp, label=f"ecc = {ecc:.1f}", **plt_args)
#     axes['hc'].plot(ta, hc, label=f"ecc = {ecc:.1f}", **plt_args)
#     wfm = hp - 1j * hc
#     axes['abs'].plot(ta, np.abs(wfm), label=f"ecc = {ecc:.1f}", **plt_args)
#
# axes['hp'].set_title("h plus")
# axes['hc'].set_title("h cross")
# axes['abs'].set_title("abs")
# axes['abs'].legend()
# axes['hp'].set_xlim(3.7, 3.85)
# axes['hc'].set_xlim(3.7, 3.85)
# axes['abs'].set_xlim(3.7, 3.85)
# fig.savefig("TEOB_pycbc_test_HM.png")

fig, axes = plt.subplots(9, 1, figsize=(17, 30))
for ecc in [0.0, 0.1, 0.2, 0.3, 0.4]:
    param['eccentricity'] = ecc
    plt_args = {'alpha': 0.45, 'lw': 0.8}
    for idx, mode in enumerate(higher_mode_arr):
        wfm_arg = {'delta_t': 1/sampling_freq, 'f_low': 10, 'ecc_freq': 1, "mode_array": [mode]}

        hp, hc = teob_source.teobresums_td(ta, **param, **wfm_arg)
        wfm = hp + 1j * hc
        # axes[idx].plot(ta, np.abs(wfm), label=f"ecc = {ecc}", **plt_args)
        axes[idx].plot(ta, hp, label=f"ecc = {ecc}", **plt_args)

    wfm_arg = {'delta_t': 1/sampling_freq, 'f_low': 10, 'ecc_freq': 1, "mode_array": higher_mode_arr}
    hp, hc = teob_source.teobresums_td(ta, **param, **wfm_arg)
    wfm = hp + 1j * hc
    axes[7].plot(ta, np.abs(wfm), label=f"ecc = {ecc}", **plt_args)
    axes[8].plot(ta, hp, label=f"ecc = {ecc}", **plt_args)

for idx, mode in enumerate(higher_mode_arr):
    axes[idx].set_title(str(mode))
    axes[idx].set_xlim(3.7, 3.85)
axes[7].set_title("abs(hp +ihc)")
axes[8].set_title("hplus")
axes[7].set_xlim(3.7, 3.85)
axes[8].set_xlim(3.7, 3.85)
axes[0].legend()
fig.suptitle(f"abs(hp + ihc)")
fig.savefig(f"TEOB_pycbc_test_modes_ecc_0to3_outallmodes_hmodes.pdf", bbox_inches="tight")
