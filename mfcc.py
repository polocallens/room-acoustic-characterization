# mfcc_utils

import librosa

from scipy import stats, signal
from scipy.io import wavfile
import glob 
import pickle
from tqdm import tqdm
import os
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (octave_low, octave_high, third_low, third_high)

import numpy as np

#---------------------------------------------------------------------------------
def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.abs(dataset).max(axis=0)

#------------------------------------------------------------------------------------------
def compute_norm_mfcc(signal,fs,n_bands,hop_len,n_fft,window_size):
    signal = feature_normalize(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc= n_bands,hop_length=hop_len, n_fft=n_fft)
    mfcc = librosa.util.fix_length(mfcc, window_size, axis=1, mode='wrap') #Reshape to windom size
    mfcc = librosa.util.normalize(mfcc, axis=1)
    return mfcc

#------------------------------------------------------------------------------------------
def compute_norm_mfcc_2(signal,fs):
    signal = feature_normalize(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs)
    #mfcc = librosa.util.fix_length(mfcc, window_size, axis=1, mode='wrap') #Reshape to windom size
    mfcc = librosa.util.normalize(mfcc, axis=1)
    return mfcc
#------------------------------------------------------------------------------------------
def compute_music_mfccs(music_dir,output_folder,window_size = 645, mfcc_bands = 40, mfcc_degree = 0,
                       chunk_duration_s = 30):
    
    music_list = sorted(glob.glob(music_dir + '*.wav'))
    folder_name = os.path.split(os.path.split(music_list[0])[0])[1]

    out_folder_name = output_folder + "mfcc_from_" + folder_name
    try:
        os.mkdir(out_folder_name)
    except:
        print("MFCC folder already exist")
        
    out_folder_name = out_folder_name + "/" + str(mfcc_degree) 
    
    try:
        os.mkdir(out_folder_name)
    except:
        print("MFCC folder already exist")
        
    for music_name in tqdm(music_list):
        filename = os.path.splitext(os.path.basename(music_name))[0]

        m_sr, music =  wavfile.read(music_name)
        
        mfcc = compute_norm_mfcc(signal = music,
                                 fs = m_sr,
                                 n_bands = mfcc_bands,
                                 hop_len=512,
                                 n_fft = 1024,
                                 window_size = window_size)
            
        if mfcc_degree == 1:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc = np.dstack((mfcc, mfcc_delta))
            
        if mfcc_degree == 2:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.dstack((mfcc, mfcc_delta, mfcc_delta2))
        
        with open(out_folder_name + "/" + filename + ".pkl", "wb") as f:
            pickle.dump(mfcc, f)
        f.close()
        
#------------------------------------------------------------------------------------------