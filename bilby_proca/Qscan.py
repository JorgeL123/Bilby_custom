#! /usr/bin/env python3
# If it does not work on your env, you may try to first source this env:
# source /home/samson.leong/miniconda3/bin/activate; conda activate --stack igwn-py37
import numpy as np
import matplotlib.pyplot as plt
from ligo.gracedb.rest import GraceDb
from gwpy.timeseries import TimeSeries

from utils import max_overlap, infft_from_ifo, njit_2nd_finite_diff
from qscan_utils import initiate_figure, plot_qspec, proca_wfm, bbh_wfm

from bilby import gw
from bilby.gw.detector import InterferometerList, PowerSpectralDensity

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/ %H:%M:%S', level=logging.INFO)

######################################################################
# Waveform Parameters 
duration = 4.
sampling_frequency = 1024.
delta_t = 1./sampling_frequency
######################################################################

######################################################################
# Event info
event = ['S190521g', 'S200114f'][1]
low_event = event.lower()
event_char = event[5:]
logging.info(f"Now we are running on event {event}.")

# Obtain geocent time from GraceDB
client = GraceDb()
event_dict = client.superevent(event).json()
geo_time = event_dict['t_0']
geo_time = 1263002916.25
ifo_list = event_dict['preferred_event_data']['instruments'].split(",")
ifo_list = ['L1']
# ifo_list.remove('V1')

# The paths to everything
indir = f'/home/samson.leong/public_html/ProcaStar/pBilby/ProcaRuns/{event}/Qscan/source/'
outdir = f'/home/samson.leong/public_html/ProcaStar/pBilby/ProcaRuns/{event}/Qscan/'
Strain_NR_path = '/home/juan.calderonbustillo/bilby/bilby_studies/HeadOn/CorrectedSimulations/strain_'
Psi4_NR_path = '/home/juan.calderonbustillo/bilby/bilby_studies/HeadOn/Psi4SimulationsScaled/'

info_dict = {
        "S190521g": {
            'proca_strain': {
                "mass_ratio": "reco15_eco9",
                "result_path": indir + "21g_reco15_eco9.json",
                "label": r'Proca strain (reco15, eco9)'},
            'proca_psi4': {
                "mass_ratio": "reco57_eco19",
                "result_path": indir + "21g_P_reco57_eco19.json",
                "label": r'Proca Psi4 (reco57, eco19)'},
            'bbh_strain': {
                "result_path": indir + "21g_NRSur4096_com_invq_q4.json",
                "label": r'BBH strain (Comoving, inv. q, Q < 4)'},
            },
        "S200114f": {
            'proca_strain': {
                "mass_ratio": "reco45_eco13",
                "result_path": indir + "14f_reco45_eco13.json",
                "label": r'Proca strain (reco45, eco13)'},
            'proca_psi4': {
                "mass_ratio": "reco41_eco20",
                "result_path": indir + "14f_P_reco41_eco20.json",
                "label": r'Proca Psi4 (reco41, eco20)'},
            'bbh_strain': {
                "result_path": indir + "14f_NRSur4096_com_invq_q4.json",
                "label": r'BBH, strain'},
            },
        }[event]

## Find the json file containing the boson mass results
key_list = list(info_dict.keys())
# print(key_list)

wfm_arg = {"sampling_frequency": 1024, "geocent_time": geo_time, "duration": 4}
proca_strain, proca_strain_param = proca_wfm(
    info_dict["proca_strain"]["mass_ratio"], info_dict["proca_strain"]["result_path"], 
    Strain_NR_path, with33mode=True, **wfm_arg)
proca_psi4, proca_psi4_param = proca_wfm(
    info_dict["proca_psi4"]["mass_ratio"], info_dict["proca_psi4"]["result_path"], 
    Psi4_NR_path, with33mode=True, psi4=True, **wfm_arg)
bbh_strain, bbh_strain_param = bbh_wfm(
    info_dict["bbh_strain"]["result_path"], wfm_approx="NRSur7dq4", **wfm_arg)

wfm_pairlist = [
    (proca_strain, proca_strain_param),
    (proca_psi4, proca_psi4_param),
    (bbh_strain, bbh_strain_param),
]
######################################################################

######################################################################
## Strain, Psi4, and PSDs
strain_paths, psi4_paths, strain_psd_paths, psi4_psd_paths = [], [], [], []
for ifo in ifo_list:
    git_repo = '/home/samson.leong/Projects/ProcaStar/proca-star-pe/production_code/'
    suffix = "txt" if event_char == "14f" else "dat"
    strain_paths.append(f'{git_repo}strain_data/{event_char}_{ifo}_1024.gwf')
    psi4_paths.append(f'{git_repo}strain_data/{event_char}_{ifo}_1024_2diff.gwf')
    strain_psd_paths.append(f'{git_repo}psd_files/{event_char}_{ifo}_PSD.{suffix}')
    psi4_psd_paths.append(f'{git_repo}psd_files/{event_char}_{ifo}_Psi4_PSD.dat')
######################################################################

# Access strain data using gwpy
start_t, end_t = geo_time - duration/2, geo_time + duration/2
xlim = (start_t, end_t)
pair = (1.1, 2.7)
strain_ifos = InterferometerList(ifo_list)
s_fig, s_axes, s_cbax = initiate_figure(key_list, ifo_list=ifo_list, info_dict=info_dict, xlim=pair)
logging.info("Starting to plot Psi4 Q-spectrogram.")

for col, ifo in enumerate(strain_ifos):
    # Obtain strain data and psd for each ifo
    ifo.strain_data.set_from_frame_file(
            frame_file=psi4_paths[col], sampling_frequency=1024., duration=4, 
            channel='None', start_time=start_t)
    ifo.power_spectral_density = PowerSpectralDensity.from_power_spectral_density_file(psi4_psd_paths[col]) 

   #  for idx in range(col, len(s_axes), len(strain_ifos)):
        # Plot the Q-scan on each axis first
    cbar = plot_qspec(s_axes[range(col, len(s_axes), len(strain_ifos))],
            ifo=ifo, start_t=start_t, outseg_pair=pair)

s_fig.colorbar(cbar, cax=s_cbax, label='Normalized energy')
s_fig.suptitle(rf"{event} Detector $\Psi_4$", fontsize=25)
s_fig.savefig(outdir+"{}_psi4_det_qspecgram.png".format(event))
plt.close(s_fig)
logging.info("Finished and saved the detectors Q-spectrogram.")

# Create three figure instances, two for Qgram, the last one is simply the TD strain
p_fig, p_axes, p_cbax = initiate_figure(key_list, ifo_list=ifo_list, info_dict=info_dict, xlim=pair)
r_fig, r_axes, r_cbax = initiate_figure(key_list, ifo_list=ifo_list, info_dict=info_dict, xlim=pair)
b_fig, b_axes, _ = initiate_figure(
    key_list, ifo_list=ifo_list, info_dict=info_dict, xlim=pair,
    xlabel=r"Time $t +${} [s]".format(start_t),
    ylabel=r"Whitened strain?? $h(t)\ [\sigma]$??", nocbar=True)
nrows = len(key_list)
ncols = len(ifo_list)

was_strain = [True, False, True]

for idx, (wfm, param) in enumerate(wfm_pairlist):

    local_start_t = param["geocent_time"] - duration/2
    # Inject Proca/BBH waveform into detectors
    ifos = InterferometerList(ifo_list)
    for kdx, ifo in enumerate(ifos):
        logging.info(ifo.name)
        psd = strain_psd_paths[kdx] if was_strain[idx] else psi4_psd_paths[kdx]
        ifo.power_spectral_density = \
            PowerSpectralDensity.from_power_spectral_density_file(psd) 
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=local_start_t)
        ifo.inject_signal(waveform_generator=wfm,
                            parameters=param)
        if was_strain[idx]:
            proca_strain_2diff = njit_2nd_finite_diff(ifo.time_domain_strain, delta_t)
            ifo_temp = InterferometerList([ifo.name])[0]
            logging.info(ifo_temp.name)
            psd = psi4_psd_paths[0]
            ifo_temp.power_spectral_density = \
                    PowerSpectralDensity.from_power_spectral_density_file(psd) 
            ifo_temp.set_strain_data_from_zero_noise(
                    sampling_frequency=sampling_frequency, duration=duration,
                    start_time=local_start_t)
            gwpydata = TimeSeries(proca_strain_2diff, times=ifo.time_array)
            ifo_temp.strain_data.set_from_gwpy_timeseries(gwpydata)
            ifos[kdx] = ifo_temp

    rho, offset = max_overlap(strain_ifos, ifos, duration/2)
    rho = 0.0; offset = 0.0
    logging.info(f"rho is {rho}, offset is {offset}")

    logging.info("Starting to plot Proca result Q-spectrogram.")
    for ax, ifo in zip(p_axes[idx * ncols: idx * ncols + ncols], ifos):
        cbar = plot_qspec([ax], ifo=ifo, start_t=local_start_t, outseg_pair=pair, offset=offset)

    logging.info("Starting to plot Residue Q-spectrogram.")
    for ax, ax2, ifo, sifo in zip(r_axes[idx * ncols: idx * ncols + ncols],
                                  b_axes[idx * ncols: idx * ncols + ncols], ifos, strain_ifos):
        ta = sifo.time_array - local_start_t
        ifo_psi4 = infft_from_ifo(sifo)
        proca_psi4 = infft_from_ifo(ifo)

        ratio = np.max(ifo_psi4) / np.max(proca_psi4)

        proca_psi4_shift = np.roll(proca_psi4, int(offset / delta_t))
        residue = ifo_psi4 - proca_psi4_shift#  * ratio

        np.savetxt(f"{outdir}/{key_list[idx]}_{ifo.name}_residue_ts.txt",
                   np.stack([ta, residue], axis=-1))

        cbar = plot_qspec([ax], timearray=ta, strain=residue, start_t=local_start_t, outseg_pair=pair)
        ax2.plot(ta, ifo_psi4)
        ax2.plot(ta, proca_psi4_shift)
        ax2.set_xlim(geo_time - start_t - 0.25, geo_time - start_t + 0.10)

p_fig.colorbar(cbar, cax=p_cbax, label='Normalized energy')
p_fig.suptitle("{} Proca result".format(event), fontsize=25)
p_fig.savefig(outdir+"{}_proca_psi4_qspecgram.png".format(event))
plt.close(p_fig)
logging.info("Finished and saved Proca result Q-spectrogram.")

r_fig.colorbar(cbar, cax=r_cbax, label='Normalized energy')
r_fig.suptitle(f"{event} Residue: Psi4 data - Proca / BBH result", fontsize=25)
r_fig.savefig(outdir+"{}_residue_qspecgram.png".format(event))

b_fig.suptitle(f"{event} Time domain waveform", fontsize=25)
b_fig.savefig(outdir+"{}_td_wfm.png".format(event))
logging.info("Finished and saved the Residue Q-spectrogram.")
