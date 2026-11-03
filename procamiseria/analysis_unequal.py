import numpy as np
import pandas as pd
from glob import glob
import pickle
from scipy.interpolate import interp1d, griddata, RectBivariateSpline
from scipy.optimize import curve_fit


class SymInterp2D:

    def __init__(self, x, y, z):
        x_fine = np.linspace(x.min(), x.max(), 100)
        fine_grid = griddata((np.concatenate([x, y]), np.concatenate([y, x])), np.concatenate([z, z]),
                             (x_fine[None, :], x_fine[:, None]), method='cubic')
        self.interp_real = RectBivariateSpline(x_fine, x_fine, np.real(fine_grid))
        if isinstance(z[0], complex):
            self.complex = True
            self.interp_imag = RectBivariateSpline(x_fine, x_fine, np.imag(fine_grid))
        else:
            self.complex = False

    def __call__(self, x, y):
        if self.complex:
            return self.interp_real(x, y)[0, 0] + 1j * self.interp_imag(x, y)[0, 0]
        return self.interp_real(x, y)[0, 0]


class DataLoader:

    def __init__(self, main_path, equal_mass = False):
        self.main_path = main_path
        self.equal_mass = equal_mass
        self.rex = 100
        self.series_data, self.omega_dict = self.slurp()
        self.lm_avail = tuple(self.series_data.lm.unique())

    def gaussian(self, x, A, x0, s):
        return A * np.exp(-((x - x0) / s) ** 2)

    def slurp(self):
        with open(self.main_path + '/Models.txt') as file:
            omega_dict = {int(line.split()[0][3:]): float(line.split()[-1]) for line in file}

        if self.equal_mass:
            series_paths = glob(self.main_path + f'reco0_eco*_mp_Psi4_l*_m*_r{self.rex:.2f}.asc')
        else:
            series_paths = glob(self.main_path + f'reco*_eco*_mp_Psi4_l*_m*_r{self.rex:.2f}.asc')
        series_data = pd.DataFrame()

        for path in series_paths:
            l = path.split('/')[-1].split('_')[-3][1:]
            m = path.split('/')[-1].split('_')[-2][1:]
            lm = l + m

            try:
                eco1 = int(path.split('/')[-1].split('_')[0][4:])
                eco2 = int(path.split('/')[-1].split('_')[1][3:])
                if eco1 == 0:
                    eco1 = eco2
                if omega_dict[eco2] > omega_dict[eco1]:
                    eco1, eco2 = eco2, eco1
            except ValueError:
                print(f"Warning: Non-integer key in filename {path.split('/')[-1]}.  Ignoring file...")
                continue

            if not (eco1 in omega_dict.keys()):
                print(f"Warning: Key {eco1} unknown in filename {path.split('/')[-1]}. Ignoring file...")
                continue
            if not (eco2 in omega_dict.keys()):
                print(f"Warning: Key {eco2} unknown in filename {path.split('/')[-1]}. Ignoring file...")
                continue

            raw_data = np.loadtxt(path)
            time = raw_data[:, 0]
            psi4 = (raw_data[:, 1] + 1j * raw_data[:, 2])

            abs_psi = np.abs(psi4)
            arg_max = np.argmax(abs_psi)
            params, _ = curve_fit(self.gaussian, time, abs_psi,
                                  p0=[abs_psi[arg_max], time[arg_max], 15])
            peak_time = params[1]

            interp_wave = interp1d(time, psi4, kind='cubic', bounds_error=False, fill_value= 0 + 0j )
            new_time = np.linspace(-170, 170, 1401)
            new_wave = interp_wave(peak_time + new_time) * (new_time > - 100)

            row_data = pd.DataFrame({'eco1': [eco1], 'eco2': [eco2], 'lm': [lm],
                                     'w1': [omega_dict[eco1]], 'w2': [omega_dict[eco2]],
                                     'peak_time': [peak_time],
                                     'time': [new_time], 'psi4': [new_wave]})

            series_data = pd.concat([series_data, row_data])

        series_data.sort_values(by=['w1', 'w2'], inplace=True)
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
        self.equal_mass = data_loader.equal_mass
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
        training_data = np.array([wf for wf in series_data_lm.psi4]).transpose()
        coefficients = np.matmul(self.encoder.basis[lm].conjugate().transpose(), training_data)

        if self.equal_mass:
            w1 = series_data_lm.w1.values
            interpolants = [interp1d(w1, coefficients[i, :], kind='cubic')
                            for i in range(self.encoder.basis_size)]
            peak_times = interp1d(w1, series_data_lm.peak_time, kind='cubic')
        else:
            w1, w2 = series_data_lm.w1.values, series_data_lm.w2.values
            interpolants = [SymInterp2D(w1, w2, coefficients[i, :])
                            for i in range(self.encoder.basis_size)]
            peak_times = SymInterp2D(w1, w2, series_data_lm.peak_time.values)

        return interpolants, peak_times

    def translate(self, w, lm):
        if self.equal_mass:
            coefficients = np.array([interp(w) for interp in self.interpolants[lm]])
        else:
            coefficients = np.array([interp(*w) for interp in self.interpolants[lm]])
        return coefficients

    def get_peak(self, w, lm):
        if self.equal_mass:
            peak_time = self.peak_times[lm](w)
        else:
            peak_time = self.peak_times[lm](*w)
        return peak_time


class SurrogateModel:

    def __init__(self):
        self.encoder = None
        self.translator = None
        self.equal_mass = None
        self.omega_dict = None
        self.lm_avail = None
        self.get_peak = None

        self.time = np.linspace(-170, 170, 1401)

    def fit(self, data_loader, basis_size=10):
        self.encoder = Encoder(data_loader, basis_size)
        self.translator = Translator(data_loader, self.encoder)
        self.equal_mass = data_loader.equal_mass
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
        self.equal_mass = self.translator.equal_mass

    def __call__(self, w, lm = '22'):

        coefficients = self.translator.translate(w, lm)
        time_shift   = self.get_peak(w, lm) - self.get_peak(w, '22')
        waveform = self.encoder.decode(coefficients, lm)
        interp_waveform = interp1d(self.time, waveform, kind='cubic', bounds_error=False, fill_value=0.0)
        new_waveform    = interp_waveform(self.time - time_shift)

        return self.time, new_waveform
