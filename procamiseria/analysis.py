import numpy as np
import pandas as pd
from glob import glob
import pickle
from scipy.interpolate import interp1d
from scipy.optimize import minimize


class DataLoader:

    def __init__(self, main_path):
        self.main_path = main_path
        self.rex = 100
        self.series_data, self.omega_dict = self.slurp()
        self.lm_avail = tuple(self.series_data.lm.unique())

    def slurp(self):
        with open(self.main_path + '/Models.txt') as file:
            omega_dict = {int(line.split()[0][3:]): float(line.split()[-1]) for line in file}

        series_paths = glob(self.main_path + f'reco0_eco*_mp_Psi4_l*_m*_r{self.rex:.2f}.asc')
        series_data = pd.DataFrame()

        for path in series_paths:
            l = path.split('/')[-1].split('_')[-3][1:]
            m = path.split('/')[-1].split('_')[-2][1:]
            lm = l + m

            eco1 = int(path.split('/')[-1].split('_')[1][3:])
            eco2 = eco1

            raw_data = np.loadtxt(path)
            time = raw_data[:, 0]
            psi4 = (raw_data[:, 1] + 1j * raw_data[:, 2])

            interp_wave = interp1d(time, psi4, kind='cubic')
            peak = np.argmax(np.abs(psi4))
            min_abs = minimize(lambda t: - np.abs(interp_wave(t)), time[peak], method='nelder-mead')
            peak_time = min_abs['x'][0]
            new_time = np.linspace(-170, 170, 1401)
            new_wave = interp_wave(peak_time + new_time) * (new_time > - 100)

            row_data = pd.DataFrame({'eco1': [eco1], 'eco2': [eco2], 'lm': [lm],
                                     'w1': [omega_dict[eco1]], 'w2': [omega_dict[eco2]],
                                     'peak_time': [peak_time],
                                     'time': [new_time], 'psi4': [new_wave]})

            series_data = pd.concat([series_data, row_data])

        series_data.sort_values(by=['w1'], inplace=True)
        series_data.reset_index(inplace=True, drop=True)

        return series_data, omega_dict


class Encoder:

    def __init__(self, data_loader, basis_size=10):
        self.lm_avail = data_loader.lm_avail
        self.basis_size = basis_size
        self.basis, self.singular_values = self.fit(data_loader.series_data)

    def fit(self, series_data):
        basis, singular_values = {}, {}
        for lm in self.lm_avail:
            basis[lm], singular_values[lm] = self.fit_lm(series_data, lm)
        return basis, singular_values

    def fit_lm(self, series_data, lm):
        series_data_lm = series_data[series_data.lm == lm]
        training_data = np.array([wf for wf in series_data_lm.psi4]).transpose()
        basis, singular_values, _ = np.linalg.svd(training_data, full_matrices=False)
        return basis[:, :self.basis_size], singular_values[:self.basis_size]

    def encode(self, waveform, lm):
        coefficients = np.matmul(self.basis[lm].conjugate().transpose(), waveform)
        return coefficients

    def decode(self, coefficients, lm):
        waveform = np.matmul(self.basis[lm], coefficients.transpose()).transpose()
        return waveform


class Translator:

    def __init__(self, data_loader, encoder):
        self.lm_avail = data_loader.lm_avail
        self.basis_size = encoder.basis_size
        self.encoder = encoder
        self.interpolants, self.peak_times = self.fit(data_loader.series_data)

    def fit(self, series_data):
        interpolants, peak_times = {}, {}
        for lm in self.lm_avail:
            interpolants[lm], peak_times[lm] = self.fit_lm(series_data, lm)
        return interpolants, peak_times

    def fit_lm(self, series_data, lm):
        series_data_lm = series_data[series_data.lm == lm]
        physical_params = series_data_lm.w1.values
        training_data = np.array([wf for wf in series_data_lm.psi4]).transpose()
        coefficients = np.matmul(self.encoder.basis[lm].conjugate().transpose(), training_data)
        interpolants = [interp1d(physical_params, coefficients[i, :], kind='cubic')
                        for i in range(self.encoder.basis_size)]
        peak_times = interp1d(physical_params, series_data_lm.peak_time, kind='cubic')
        return interpolants, peak_times

    def translate(self, w1, lm):
        coefficients = np.array([interp(w1) for interp in self.interpolants[lm]])
        return coefficients

    def get_peak(self, w1, lm):
        peak_time = self.peak_times[lm](w1)
        return peak_time


class SurrogateModel:

    def __init__(self):
        self.encoder = None
        self.translator = None
        self.omega_dict = None
        self.lm_avail = None
        self.get_peak = None

        self.time = np.linspace(-170, 170, 1401)

    def fit(self, data_loader, basis_size=10):
        self.encoder = Encoder(data_loader, basis_size)
        self.translator = Translator(data_loader, self.encoder)
        self.omega_dict = data_loader.omega_dict
        self.lm_avail   = data_loader.lm_avail
        self.get_peak = self.translator.get_peak

    def save(self, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(self.encoder, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.translator, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.omega_dict, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as inp:
            self.encoder = pickle.load(inp)
            self.translator = pickle.load(inp)
            self.omega_dict = pickle.load(inp)
        self.lm_avail = self.encoder.lm_avail
        self.get_peak = self.translator.get_peak

    def __call__(self, w1, lm = '22'):
        coefficients = self.translator.translate(w1, lm)
        waveform = self.encoder.decode(coefficients, lm)

        interp_waveform = interp1d(self.time, waveform, kind='cubic', bounds_error=False, fill_value=0.0)
        new_waveform    = interp_waveform(self.time - self.get_peak(w1, lm) + self.get_peak(w1, '22'))

        return self.time, new_waveform
