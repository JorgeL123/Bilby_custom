####################################################################################
# Below are deprecated functions and stuff
# Dependencies are not included, and could be broken.
####################################################################################

def pis4_hphc_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):


    h22 = waveform_arguments["h22"]
    h20 = waveform_arguments["h20"]
    h32 = waveform_arguments["h32"]
    h33 = waveform_arguments.get("h33", None)
    mylen = np.min([len(h22),len(h20),len(h32)])
    delta_t = waveform_arguments["delta_t"]


    t_start = - total_mass/1300. 
    # 1/1300 = 0.2/260 
    # This value has be chosen becoz when we scale to 260 Mdot, 
    # the entire pre-merger is contained in the first 0.2s, exlcuding "junk radiation".

    time = total_mass * np.transpose(h22)[0]
    times = np.arange(time[0], time[-1], delta_t)
    MD_ratio = total_mass / luminosity_distance

    # h22s = fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)

    # h20s = fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    # h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)
    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)


    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s

# We check for higher modes and add them one by one:

    if h32 is not None:
        h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)
        h3m2s = -np.conjugate(h32s)
        H += Ydirect(theta_jn, phase, 3, 2)*h32s +\
            Ydirect(theta_jn, phase, 3, -2)*h3m2s

    if h20 is not None:
        h20s = fromNRtoPhysicalUnits(h30, time, times, MD_ratio, delta_t)
        H += Ydirect(theta_jn, phase, 2, 0)*h20s

    if h33 is not None:
        h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
        h3m3s = -np.conjugate(h33s)

        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
            Ydirect(theta_jn, phase, 3, -3)*h3m3s

    hp = np.real(H)
    hc = np.imag(H)
    
    hp22 = np.real(h22s)
    hc22 = np.imag(h22s)
    
    waveform_time = np.arange(0, mylen, delta_t)

    amp22 = np.abs(h22s)
    
    gps_index = np.argmin(np.abs(time_array - geocent_time))
    
    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))
    
    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index


    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)

    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = padding(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = padding(h_cross_windowed, before=N_start, after=N_end)

    return {'plus': h_plus_padded, 'cross': h_cross_padded}


def strain_hphc_psi4_from_strain_from_mode_arr(time_array, total_mass, luminosity_distance,
                theta_jn, phase, geocent_time, **waveform_arguments):

    h22 = waveform_arguments["h22"]
    h20 = waveform_arguments["h20"]
    h32 = waveform_arguments["h32"]
    h33 = waveform_arguments.get("h33", None)
    mylen = np.min([len(h22),len(h20),len(h32)])
    delta_t = waveform_arguments["delta_t"]

    t_start = - total_mass/1300.
    # 1/1300 = 0.2/260 
    # This value has be chosen becoz when we scale to 260 Mdot, 
    # the entire pre-merger is contained in the first 0.2s, exlcuding "junk radiation".

    time = total_mass * np.transpose(h22)[0]
    times = np.arange(time[0], time[-1], delta_t)
    MD_ratio = total_mass / luminosity_distance

    h22s = fromNRtoPhysicalUnits(h22, time, times, MD_ratio, delta_t)
    h20s = fromNRtoPhysicalUnits(h20, time, times, MD_ratio, delta_t)
    h32s = fromNRtoPhysicalUnits(h32, time, times, MD_ratio, delta_t)

    h2m2s = np.conjugate(h22s)
    h3m2s = -np.conjugate(h32s)
    # h2m0s = np.conjugate(h20s)

    H = Ydirect(theta_jn, phase, 2, 2)*h22s +\
        Ydirect(theta_jn, phase, 2, 0)*h20s +\
        Ydirect(theta_jn, phase, 3, 2)*h32s +\
        Ydirect(theta_jn, phase, 2, -2)*h2m2s +\
        Ydirect(theta_jn, phase, 3, -2)*h3m2s
    if h33 is not None:
        h33s = fromNRtoPhysicalUnits(h33, time, times, MD_ratio, delta_t)
        h3m3s = -np.conjugate(h33s)

        H += Ydirect(theta_jn, phase, 3, 3)*h33s +\
            Ydirect(theta_jn, phase, 3, -3)*h3m3s

    hp = np.real(H)
    hc = np.imag(H)

    hp22 = np.real(h22s)
    hc22 = np.imag(h22s)

    waveform_time = np.arange(0, mylen, delta_t)
    amp22 = np.abs(h22s)

    gps_index = np.argmin(np.abs(time_array - geocent_time))

    template_max_index = np.argmax(amp22)
    template_max_time = waveform_time[template_max_index]
    start_index = np.argmin(np.abs(waveform_time - template_max_time - t_start))

    index_offset = template_max_index - start_index \
                    if t_start < 0 else \
                   start_index - gps_index

    start_idx = template_max_index - index_offset
    end_idx = np.min([template_max_index+len(time_array) + index_offset, len(hp)])
    window = tukey(end_idx - start_idx, alpha=0.5)

    h_plus_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_cross_windowed = np.empty(end_idx - start_idx + 1, dtype=np.float64)
    h_plus_windowed = hp[start_idx : end_idx] * window
    h_cross_windowed = hc[start_idx : end_idx] * window

    N_start = 0
    N_end = np.abs(len(h_plus_windowed) - len(time_array)) - N_start

    h_plus_padded = mypad(h_plus_windowed, before=N_start, after=N_end)
    h_cross_padded = mypad(h_cross_windowed, before=N_start, after=N_end)


    ##### Now, here, we take the second-order finite differences

    h_plus_padded_psi4 = njitfindiff2(h_plus_padded, delta_t)
    h_cross_padded_psi4 = njitfindiff2(h_cross_padded, delta_t)



    return {'plus': h_plus_padded_psi4, 'cross': h_cross_padded_psi4}


def fromNRtoPhysicalUnitsPsi4(mode,time,times,total_mass,luminosity_distance,delta_t):

    
    #time = total_mass*4.92e-6*np.transpose(mode)[0]
    strain = 100./(total_mass*4.9254184936691445e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)*(np.transpose(mode)[1]+1j*np.transpose(mode)[2])
    #strain = 100*(total_mass*4.92e-6)**2*(total_mass*SunMass*GN/SpeedOfLight**2/MegaParsec/luminosity_distance)*(np.transpose(mode)[1]+1j*np.transpose(mode)[2])
    #hinterp = scipy.interpolate.interp1d(time,strain)

    #times = np.arange(time[0],time[-1],delta_t)


    #strains = hinterp(times)

    #return np.array([times,strains])

    if len(time) != len(times):
        hinterp = interp1d(time, strain)
        strains = hinterp(times)

    return strains

