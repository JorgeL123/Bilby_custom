#! /usr/bin/env python3
# If it does not work on your env, you may try to first source this env:
# source /home/samson.leong/miniconda3/bin/activate; conda activate --stack igwn-py37
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from .source import strain_hcpc_from_dat_file, psi4_hcpc_from_dat_file
from .utils import infft, njit_2nd_finite_diff

from bilby import gw
from bilby.core.result import read_in_result as read

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/ %H:%M:%S', level=logging.INFO)

######################################################################
## Define two constant for Q-transform
tres = 3 # time resolution
fres = 2 # freq resolution

######################################################################
## Define a bunch of functionssssss for figures

def initiate_figure(key_list, ifo_list=["H1", "L1"], info_dict={},
                    xlim=(1, 3), xlabel=None, ylabel=None, nocbar=False):
    """
    Function to inititate a figure instance with n * m subplots, for
    n being the number of different waveforms; and
    m being the number of detectors
    ##########
    Parameters
    ##########
    plots: list of keys 
    """
    # Create an axis for each detector
    nrows = len(key_list)
    ncols = len(ifo_list)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols*12, nrows*3.7),
        sharex='all', subplot_kw={'xlim': xlim})
    if not nocbar:
        plt.subplots_adjust(hspace=0.15, left=0.07, right=0.9, top=0.91, bottom=0.07)
        cb_ax = fig.add_axes([0.92, 0.15, 0.034, 0.7])
    else:
        plt.subplots_adjust(hspace=0.15, left=0.07, right=0.94, top=0.91, bottom=0.07)
        cb_ax = None

    for ax, key in zip(axes, key_list * ncols):
        # ax.text(0.07, 0.9, ifo_name, horizontalalignment='left', color='C3',
        #         verticalalignment='center', transform=ax.transAxes, fontsize=16,
        #         bbox={'facecolor':'none', 'edgecolor':'C3', 'pad':6})
        ax.set_title(info_dict[key]["label"], fontsize=17)
        ax.set_xlabel("  ")

    if xlabel is None:
        xlabel = r"Time $t +${} [s]".format(xlim[0])
    if ylabel is None:
        ylabel = "Frequency [Hz]"
    for idx in range((nrows - 1) * ncols, nrows * ncols):
        axes[idx].set_xlabel(xlabel, fontsize=15)
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=15)
    return fig, axes, cb_ax


def plot_qspec(axes, ifo=None, t_res=tres, f_res=fres, outseg_pair=None,
               start_t=0, offset=None, frange=(10, 512), qrange=(4, 64),
               convert_to_psi4=False, timearray=None, strain=None):
    """ Function to plot the Q-spectrogram
    It takes either an interferometer object or a strain & time array.
    In the case of an interferometer object, it will extract the frequency domain strain and 
    convert it to time domain through infft.
    An offset would be applied to it if provided and then plot the Q-spectrogram.
    """
    if (ifo is None) and (timearray is None) and (strain is None):
        logging.error("Please at least give me an ifo or the pair of time and strain array.")
    if (timearray is not None) and (strain is not None):
        time_array = timearray
        strain_array = strain
        ifo = None
    else:
        duration = ifo.duration
        time_array = ifo.time_array - start_t
        strain_array = -1 * infft(ifo.whitened_frequency_domain_strain,
                                sampling_frequency=ifo.strain_data.sampling_frequency)\
                    / (2*np.pi*duration)

    delta_t = time_array[1] - time_array[0]
    if offset is not None:
        strain_array = np.roll(strain_array, int(offset / delta_t))

    if convert_to_psi4:
        psi4_array = njit_2nd_finite_diff(strain_array, delta_t)
        ts = TimeSeries(psi4_array, times=time_array)
    else:
        ts = TimeSeries(strain_array, times=time_array)

    logging.info("Performing Q-Transform.")
    qspecgram = ts.q_transform(
        outseg=outseg_pair, logf=True, tres=10**-t_res, fres=10**f_res,
        frange=frange, qrange=qrange, highpass=11)

    for ax in axes:
        cbar = ax._pcolormesh_array2d(qspecgram, vmin=0, vmax=60)
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.set_ylim(frange)

        ax.minorticks_off()
        ax.set_yticks([16, 32, 64, 128, 256, 512])
        ax.set_yticklabels([16, 32, 64, 128, 256, 512])
    return cbar

######################################################################
# Waveform generator utils

def proca_wfm(mass_ratio, result_path, NR_path, with33mode=True, psi4=False, **waveform_argument):
    n1 = mass_ratio.find("eco", 0)
    n2 = mass_ratio.find("eco", n1 + 1)
    e1 = mass_ratio[n1 + 3 : n2 - 1]
    e2 = mass_ratio[n2 + 3 :]
    print(e1, e2)
    if e1 == e2:
        mass_ratio = "eco" + e1
    # Suffix
    suffix = "asc" if psi4 else "dat"
    # Create the NR file paths
    strain_file_paths = [
            f"{mass_ratio}_mp_Psi4_l2_m2_r100.00.{suffix}",
            f"{mass_ratio}_mp_Psi4_l2_m0_r100.00.{suffix}",
            f"{mass_ratio}_mp_Psi4_l3_m2_r100.00.{suffix}"
            ]
    if with33mode:
        strain_file_paths.append(
                f"{mass_ratio}_mp_Psi4_l3_m3_r100.00.{suffix}"
                )

    # Find the posterior of that particular simulation
    try:
        poste = read(filename=result_path).posterior
    except OSError:
        logging.error("Cannot find this: " + result_path)

    # Locate the max logL entry
    index = np.argmax(poste['log_likelihood'])

    # Extract the necessary info from waveform_arguments
    sampling_frequency = waveform_argument.get("sampling_frequency", 1024)
    geo_time = waveform_argument.get("geocent_time", None)
    duration = waveform_argument.get("duration", 4)

    # Prepare the injection paramters
    param_names = ['total_mass', 'phase', 'theta_jn', 'luminosity_distance']
    pair_list = [(name, poste[name][index]) for name in param_names]
    waveform_parameters = dict(pair_list)
    waveform_parameters['delta_t'] = 1. / sampling_frequency
    # waveform_parameters['t_start'] = - waveform_parameters['total_mass'] / 2600
    try:
        time = poste["geocent_time"][index] 
    except KeyError:
        print("Reverting back to input geo_time")
        time = geo_time
    waveform_parameters['geocent_time'] = time
    waveform_parameters['strain_paths_set'] = strain_file_paths
    waveform_parameters['NR_path'] = NR_path

    # Prepare the waveform generator
    waveform = gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        time_domain_source_model=\
                strain_hcpc_from_dat_file if not psi4 else psi4_hcpc_from_dat_file,
        waveform_arguments=waveform_parameters)

    # Add the extrinsic parameters to make a full injection set
    extrinsic_set = ['psi', 'ra', 'dec']
    expair_list = {name: poste[name][index] for name in extrinsic_set}
    injection_parameters = dict(**waveform_parameters, **expair_list)

    return waveform, injection_parameters


def bbh_wfm(result_path, wfm_approx="IMRPhenomXPHM", **waveform_argument):
    # Find the posterior of that particular simulation
    try:
        result = read(filename=result_path)
        poste = result.posterior
    except OSError:
        logging.error("Cannot find this: " + result_path)

    # Locate the max logL entry
    index = np.argmax(poste['log_likelihood'])

    # Extract the necessary info from waveform_arguments
    sampling_frequency = waveform_argument.get("sampling_frequency", 1024)
    geo_time = waveform_argument.get("geocent_time", None)
    duration = waveform_argument.get("duration", 4)

    # Prepare the injection paramters
    param_names = result.search_parameter_keys
    # ref_freq = result.waveform_arguments['reference_frequency']
    # min_freq = result.waveform_arguments['minimum_frequency']
    pair_list = [(name, poste[name][index]) for name in param_names]
    waveform_parameters = dict(pair_list)
    waveform_parameters['geocent_time'] = geo_time
    waveform_arguments = dict(waveform_approximant=wfm_approx,
                              reference_frequency=20,
                              minimum_frequency=8)

    # Prepare the waveform generator
    waveform = gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments)

    injection_parameters = waveform_parameters

    return waveform, injection_parameters
