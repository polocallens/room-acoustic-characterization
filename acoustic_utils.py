#Acoustic utilities
from tqdm import tqdm
import scipy.io.wavfile as wavfile
from scipy import stats, signal
import os
import pickle

import glob
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
import numpy as np
#---------------------------------------------------------------------------------
def t60_impulse(raw_signal,fs, bands, rt='t30'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """

    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    
    
    if bands is not None:
        t60 = np.zeros(bands.size)
    else:
        t60 = np.zeros(1)

    for band in range(len(t60)):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60


#---------------------------------------------------------------------------------
def t60_impulse_avg(raw_signal,fs, rt='t30'):  # pylint: disable=too-many-locals
    """
    Average reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0


    # Filtering signal
    abs_signal = np.abs(raw_signal) / np.max(np.abs(raw_signal))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60
#------------------------------------------------------------------------------------------


"""def rir2t60(rir_dir, output_folder):
    t60s = []
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    try:
        os.mkdir(output_folder + "t60")
    except:
        print("t60 folder already exist")
        
    #Process all RIRs and save outputs as pkl
    for rir_file in tqdm(rir_list):
        try:
            rir_sr, rir = wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 

        bands = acoustics.bands.third(500,8000)

        t60 = t60_impulse(rir, rir_sr,  bands, rt='t30')
        t60s.append(t60)
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        with open(output_folder + "t60/" + filename + ".pkl", "wb") as f:
            pickle.dump(t60, f)
        f.close()
    return t60s
"""
#------------------------------------------------------------------------------------------


def clarity_avg(time, signal, fs):
    """
    Clarity :math:`C_i` determined from an impulse response.

    :param time: Time in miliseconds (e.g.: 50, 80).
    :param signal: Impulse response.
    :type signal: :class:`np.ndarray`
    :param fs: Sample frequency.
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`

    """
    h2 = signal**2.0
    t = int((time / 1000.0) * fs + 1)
    c = 10.0 * np.log10((np.sum(h2[:t]) / np.sum(h2[t:])))
    return c

#------------------------------------------------------------------------------------------
def rir2clarity(rir_dir, output_folder,time):
    #Process all RIRs and save outputs as pkl
    clarities = []
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    try:
        os.mkdir(output_folder + "c" + str(time))
    except:
        print("Clarity folder already exist")
        
    for rir_file in tqdm(rir_list):
        try:
            rir_sr, rir =  wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 

        bands = acoustics.bands.third(500,8000)
        
        clarity = acoustics.room.clarity(time, rir, rir_sr, bands)
        clarities.append(clarity)
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        with open(output_folder + "c" + str(time) + "/" + filename + ".pkl", "wb") as f:
            pickle.dump(clarity, f)
        f.close()
    return(clarities)

#------------------------------------------------------------------------------------------
"""def rir2drr(rir_dir, output_folder):
    drrs = []
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    a=1
    try:
        os.mkdir(output_folder + "drr")
    except:
        print("drr folder already exist")
        
    #Process all RIRs and save outputs as pkl
    for rir_file in tqdm(rir_list):
        try:
            rir_sr, rir =  wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 

        onset = np.argmax(np.abs(rir))
        direct_range = int(5**-3 * rir_sr)
        
        # RIR decomposition
        direct = rir[:onset + direct_range]
        reverb = rir[onset + direct_range + 1:]
        epsilon = 1e-12


        # DDR calculation

        DRR = (np.sum(np.square(direct)) + epsilon) / (np.sum(np.square(reverb)) + epsilon)
        #print(f'drr before log = {DRR}')
        DRR = 10*np.log10(DRR)
        drrs.append(DRR)
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        with open(output_folder + "drr/" + filename + ".pkl", "wb") as f:
            pickle.dump(DRR, f)
        f.close()
    return drrs
"""
#------------------------------------------------------------------------------------------

def drr(signal, fs, bands=None, debug = False):
    """
    Clarity :math:`C_i` determined from an impulse response.
    :param time: Time in miliseconds (e.g.: 50, 80).
    :param signal: Impulse response.
    :type signal: :class:`np.ndarray`
    :param fs: Sample frequency.
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`
    """
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    direct_range = int(2.5e-3 * fs) #2.5ms window around the peak is what is used by ACE challenge
    
    if bands is not None:
        #print('\n\nall\n\n')
        drr = np.zeros(bands.size)
        for band in range(bands.size):
            filtered_signal = bandpass(signal, low[band], high[band], fs, order=8)

            h2 = np.abs(filtered_signal) * np.abs(filtered_signal)

            onset = np.argmax(np.abs(h2))

            if onset > direct_range:
                direct = h2[onset - direct_range : onset + direct_range]
                reverb = np.concatenate((h2[:onset - direct_range - 1], h2[onset + direct_range + 1:]))
            else:
                direct = h2[:2*direct_range]
                reverb = h2[2*direct_range+1 :]

            drr[band] = 10.0 * np.log10((np.sum(np.abs(direct)) / np.sum(np.abs(reverb))))
            
    else:
        h2 = np.abs(signal) * np.abs(signal)

        onset = np.argmax(np.abs(h2))

        if onset > direct_range:
            direct = h2[onset - direct_range : onset + direct_range]
            reverb = np.concatenate((h2[:onset - direct_range - 1], h2[onset + direct_range + 1:]))
        else:
            direct = h2[:2*direct_range]
            reverb = h2[2*direct_range+1 :]

        drr = 10.0 * np.log10((np.sum(np.abs(direct)) / np.sum(np.abs(reverb))))
        
        if debug:
            print(f'Onset : {onset}')
            print(f'direct : {direct}')
            print(f'reverb : {reverb}')
            print(f'drr before log : {((np.sum(direct) / np.sum(reverb)))}')
            print(f'drr final : {drr}')


            
    return drr

#------------------------------------------------------------------------------------------
def rir2drr(rir_dir, output_folder = None, bands=None):
    
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    
    if output_folder is not None:
        try:
            os.mkdir(output_folder + "drr")
        except:
            print("drr folder already exist")
        
    if bands is not None:
        drrs = np.empty((len(rir_list),len(bands)))
    else:
        drrs = np.empty(len(rir_list))
        
    #Process all RIRs and save outputs as pkl
    for i, rir_file in enumerate(tqdm(rir_list)):
        #print(f'RIR --> {rir_file}')
        try:
            rir_sr, rir =  wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 
        
        drrs[i] = drr(rir, rir_sr, bands)
        
        if np.isnan(drrs[i]):
            print(f'isnan : {rir_file}\n')
            print(drrs[i])
        #print(drrs)
        
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        
        if output_folder is not None :
            with open(output_folder + "drr/" + filename + ".pkl", "wb") as f:
                pickle.dump(drrs[i], f)
            f.close()
        
    #print(f'drr after log = {drrs}\n')
    return drrs

#------------------------------------------------------------------------------------------
def rir2t60(rir_dir, output_folder,bands = acoustics.bands.third(500,8000)):
    #t60s = []
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    try:
        os.mkdir(output_folder + "t60")
    except:
        print("t60 folder already exist")
        
        
    if bands is not None:
        t60s = np.empty((len(rir_list),len(bands)))
    else:
        t60s = np.empty(len(rir_list))
        
    #Process all RIRs and save outputs as pkl
    for i,rir_file in enumerate(tqdm(rir_list)):
        try:
            rir_sr, rir = wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 

        #bands = acoustics.bands.third(500,8000)

        t60s[i] = t60_impulse(rir, rir_sr,  bands, rt='t30')
        
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        with open(output_folder + "t60/" + filename + ".pkl", "wb") as f:
            pickle.dump(t60s[i], f)
        f.close()
    return t60s

#------------------------------------------------------------------------------------------
"""def rir2t60(rir_dir, output_folder):
    t60s = []
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    try:
        os.mkdir(output_folder + "t60")
    except:
        print("t60 folder already exist")
        
    #Process all RIRs and save outputs as pkl
    for rir_file in tqdm(rir_list):
        try:
            rir_sr, rir = wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            return None 

        bands = acoustics.bands.third(500,8000)

        t60 = t60_impulse(rir, rir_sr,  bands, rt='t30')
        t60s.append(t60)
        filename = os.path.splitext(os.path.basename(rir_file))[0]
        with open(output_folder + "t60/" + filename + ".pkl", "wb") as f:
            pickle.dump(t60, f)
        f.close()
    return t60s
"""