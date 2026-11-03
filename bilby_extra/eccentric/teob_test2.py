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

fig, axes = plt.subplot_mosaic([['hp', 'hc'],['abs', 'abs']], figsize=(20, 10))
# plt_args = {'alpha': 0.55, 'lw': 0.9}
# # fig, axes = plt.subplots(1, 2, figsize=(17, 6))
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

for ecc in [0.0, 0.1, 0.2]:
    param['eccentricity'] = ecc
    fig, axes = plt.subplot_mosaic([['hp', 'hc'],['hpphc', 'hpphc'],['absp', 'absp'], ['absm', 'absm']], figsize=(20, 18))
    plt_args = {'alpha': 0.45, 'lw': 0.8}
    for mode in higher_mode_arr:
        wfm_arg = {'delta_t': 1/sampling_freq, 'f_low': 10, 'ecc_freq': 1, "mode_array": [mode]}

        param['eccentricity'] = 0.0
        hp, hc = teob_source.teobresums_td(ta, **param, **wfm_arg)
        axes['hp'].plot(ta, hp, label=f"mode = {mode}", **plt_args)
        axes['hc'].plot(ta, hc, label=f"mode = {mode}", **plt_args)
        wfm = hp - 1j * hc
        axes['hpphc'].plot(ta, hp*hp + hc*hc, label=f"mode = {mode}", **plt_args)
        # axes['hpmhc'].plot(ta, hp*hp - hc*hc, label=f"mode = {mode}", **plt_args)
        axes['absp'].plot(ta, np.abs(hp - 1j * hc), label=f"mode = {mode}", **plt_args)
        axes['absm'].plot(ta, np.abs(hp + 1j * hc), label=f"mode = {mode}", **plt_args)

    axes['hp'].set_title("h plus")
    axes['hc'].set_title("h cross")
    axes['hpphc'].set_title("hp*hp + hc*hc")
    axes['absp'].set_title("abs(hp + i hc)")
    axes['absm'].set_title("abs(hp - i hc)")
    axes['absp'].legend()
    for ax in axes:
        axes[ax].set_xlim(3.7, 3.85)
    fig.suptitle(f"ecc = {ecc}")
    fig.savefig(f"TEOB_pycbc_test_modes_ecc{int(ecc*10)}.pdf")
