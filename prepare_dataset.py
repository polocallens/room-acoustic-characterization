# Load various imports 
import pandas as pd
import os
import librosa
import glob
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

import pickle
from tqdm import tqdm

from keras.layers import Permute
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.recurrent import GRU, LSTM
from tensorflow.keras.utils import Sequence

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, CuDNNLSTM, BatchNormalization
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Reshape, TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
from keras.models import load_model
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
import datetime

import scipy.io.wavfile as wavfile

from subprocess import call

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

from scipy import stats, signal
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (octave_low, octave_high, third_low, third_high)

from argparse import ArgumentParser


#from keras.callbacks import TensorBoard
from time import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')


#---------------------------------------------------------------------------------
from mfcc import *
from acoustic_utils import *
    
    
    
# Parameters

def parse_args():
    parser = ArgumentParser(description='Data preparation')
    
    parser.add_argument(
        '-musicDir', '--musicDir',
        type=str, default='Datasets/Music/smaller2_fma_all_wav/',
        help='Music folder.'
    )
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, default='Datasets/RIRs/440_normed/',
        help='Rir folder.'
    )

    parser.add_argument(
        '-outDir', '--outDir',
        type=str, default='default_dataset/',
        help='Output dataset folder.'
    )
    
    parser.add_argument(
        '-window_size', '--window_size',
        type=int, default=645,
        help='Number of timesteps.'
    )
    
    parser.add_argument(
        '-mfcc_bands', '--mfcc_bands',
        type=int, default=40,
        help='Number of mfcc bands (along freq axis)'
    )
    
    parser.add_argument(
        '-mfcc_degree', '--mfcc_degree',
        type=int, default=0,
        help='Using additional delta mfcc layer (0,1 or 2).'
    )
    
    parser.add_argument(
        '-chunk_duration_s', '--chunk_duration_s',
        type=int, default=30,
        help='Music sample Duration in seconds'
    )
    
    parser.add_argument('--recompute', dest='recompute', action='store_true')
    parser.add_argument('--no-recompute', dest='recompute', action='store_false')
    parser.set_defaults(recompute=True)
    
    return parser.parse_args()

#---------------------------------------------------------------------------------
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#---------------------------------------------------------------------------------
def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.abs(dataset).max(axis=0)

#------------------------------------------------------------------------------------------
def combinations(music_list,rir_list):
    combinations = np.stack(np.meshgrid(music_list,rir_list),-1).reshape(-1,2)
    
    indexes = np.arange(len(combinations))
    
    music_names = [combinations[k][0] for k in indexes]
    rir_names = [combinations[k][1] for k in indexes]
    return music_names , rir_names

#------------------------------------------------------------------------------------------
def make_convolved_dataset(music_dir, rir_dir, output_folder, 
                           window_size = 645, mfcc_bands = 40, mfcc_degree = 0,
                          chunk_duration_s = 30):
    
    ac_params = ['t60','c50','c80','drr']
    
    music_list = sorted(glob.glob(music_dir + '*.wav'))
    rir_list = sorted(glob.glob(rir_dir + '*.wav'))
    
    music_names, rir_names = combinations(music_list,rir_list)

    try : 
        os.mkdir(output_folder)
    except:
        print('Output folder already exists')    
    try : 
        os.mkdir(output_folder+"X")
    except:
        print('X folder already exists')

    try : 
        os.mkdir(output_folder+"y")
    except:
        print('y folder already exists')
        
    for param in ac_params:
        try : 
            os.mkdir(output_folder + "y/" + param)
        except:
            print(f'{param} folder already exists')   
        
    folder_name = os.path.split(os.path.split(music_list[0])[0])[1]
    out_folder_name = output_folder + "mfcc_from_" + folder_name + "/" + str(mfcc_degree) 
    print('tot_number_comb:', len(music_names))

    for i,(music_path, rir_path) in tqdm(enumerate(zip(music_names,rir_names))):
        
        #Extract music and rir names 
        
        music_name = os.path.splitext(os.path.basename(music_path))[0]
        rir_name = os.path.splitext(os.path.basename(rir_path))[0]
        
        #Check if file already exists to gain time in case of interruption
        if os.path.exists(output_folder + "X/" + str(i) + ".pkl"):
            continue

        else:
            #Load and normalize audio files
            #print(f'path : {music_path}\n')
            
            try: 
                #music, m_sr = librosa.load(music_path, res_type='kaiser_fast')
                m_sr, music = wavfile.read(music_path)
                music = feature_normalize(music)
                #music = music[:chunk_duration_s*m_sr]

            except Exception as e:
                print("Error encountered while parsing music file: ", music_path)
                return None 

            try:
                #rir, rir_sr = librosa.load(rir_path, res_type='kaiser_fast')
                rir_sr, rir = wavfile.read(rir_path)
            except:
                print("Error encountered while parsing rir file: ",rir_path)
                return None 

        
            #Apply convolution, full mode to avoid introducing a time shift in the generated reverberant signal
            #print(f'\n\n music shape : {music.shape} \n rir shape : {rir.shape}\n \n')
            music_rev = signal.fftconvolve(music, rir, mode="full")
            #Cut the tail to keep music and reverberant music the same length
            #Normalize after convolution
            music_rev = feature_normalize(music_rev)

            #print(f'music shape : {len(music)} \n {music.shape}')
            music_rev = music_rev[:len(music)]
            
            #Compute MFCC
            #print(f'music name = {music_name} \nrir name : {rir_name}')
            mfcc_rev = librosa.feature.mfcc(y=music_rev, sr=m_sr, n_mfcc= mfcc_bands,hop_length=512, n_fft=1024)
            mfcc_rev = librosa.util.fix_length(mfcc_rev, window_size, axis=1, mode='wrap') #Reshape to windom size
            mfcc_rev = librosa.util.normalize(mfcc_rev, axis=1)
            
            #Degree of mfcc
            if mfcc_degree == 1:
                mfcc_delta_rev = librosa.feature.delta(mfcc_rev)
                mfcc_rev = np.dstack((mfcc_rev, mfcc_delta_rev))

            if mfcc_degree == 2:
                mfcc_delta_rev = librosa.feature.delta(mfcc_rev)
                mfcc_delta2_rev = librosa.feature.delta(mfcc_rev, order=2)
                mfcc_rev = np.dstack((mfcc_rev, mfcc_delta_rev, mfcc_delta2_rev))

            #Concatenate music mfcc with reverberant music mfcc
            with open(out_folder_name + "/" + music_name + ".pkl", "rb") as f:
                if mfcc_degree == 0:
                    mfcc = pickle.load(f)
                    X = np.dstack((mfcc, mfcc_rev))
                elif mfcc_degree == 1:
                    [mfcc, mfcc_delta] = pickle.load(f)
                    X = np.dstack((mfcc, mfcc_rev, mfcc_delta, mfcc_delta_rev))
                elif mfcc_degree == 2:
                    [mfcc, mfcc_delta, mfcc_delta2] = pickle.load(f)
                    X = np.dstack((mfcc, mfcc_rev, mfcc_delta, mfcc_delta_rev, mfcc_delta2, mfcc_delta2_rev))
                else: 
                    print("Error : mfcc has to be between 0 and 2")
                    return None
                f.close()
                
            #Save X files
            with open(output_folder + "X/" + str(i) + ".pkl", "wb") as f:
                pickle.dump(X, f)
            f.close()

            #Load y true files and save it to y with the right name
            #t60
            
            for param in ac_params:

                with open(output_folder + param + "/" + rir_name + ".pkl", "rb") as f:
                    y = pickle.load(f)
                f.close()

                with open(output_folder + "y/" + param + "/" + str(i) + ".pkl", "wb") as f:
                    pickle.dump(y, f)
                f.close()

            

#------------------------------------------------------------------------------------------
def normalize_files(file_dir):
    
    print(f'------- Normalizing directory : {file_dir} -------')
    file_list = sorted(glob.glob(file_dir + '*'))
    
    out_dir = file_dir.strip("/") + '_sox_normed/'
    try : 
        os.mkdir(out_dir)
    except:
        print('Output folder already exists')
        
    for file in tqdm(file_list):
        filename = os.path.split(file)[1]
        call('sox -G ' + file + ' -r 22050 -c 1 -b 16 ' + out_dir + '/' + filename,shell=True)

#------------------------------------------------------------------------------------------
def chunk_audio_files(music_dir,chunk_duration_s):
    print(f'------- Chunk audio directory : {music_dir} -------')
    music_list = sorted(glob.glob(music_dir + '*'))
    
    out_dir = music_dir.strip("/") + '_'+ str(chunk_duration_s) + 's/'
    
    try : 
        os.mkdir(out_dir)
    except:
        print('Output folder already exists')  
        
    for file in tqdm(music_list):
        filename = os.path.split(file)[1]
        call('sox ' + file + ' ' + out_dir + '/' + filename + ' trim 0 ' + str(chunk_duration_s),shell=True)
#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    try : 
        os.mkdir(args.outDir)
    except:
        print('Output folder already exists')  
        
    if (args.recompute==True):
        print("Normalizing music...")
        normalize_files(args.musicDir)
        normedMusicDir = args.musicDir.strip("/") + '_sox_normed/'
        print("Chunking...")
        chunk_audio_files(normedMusicDir, args.chunk_duration_s)
        chunkedNormedMusicDir = args.musicDir.strip("/") + '_sox_normed_' + str(args.chunk_duration_s) + 's/'
        
        print("Normalizing RIRs...")
        normalize_files(args.rirDir)
        normedRirDir =  args.rirDir.strip("/") + '_sox_normed/'
    
        print("Computing t60s...")
        rir2t60(normedRirDir, args.outDir)
        print("Computing c50...")
        rir2clarity(normedRirDir, args.outDir,50)
        print("Computing c80...")
        rir2clarity(normedRirDir, args.outDir,80)
        print("Computing drr...")
        rir2drr(normedRirDir, args.outDir)
        
        print("Computing music MFCCs...")
        compute_music_mfccs(chunkedNormedMusicDir,args.outDir,args.window_size, args.mfcc_bands, args.mfcc_degree, args.chunk_duration_s)

    else:
        chunkedNormedMusicDir = args.musicDir.strip("/") + '_sox_normed_' + str(args.chunk_duration_s) + 's/'
        normedRirDir =  args.rirDir.strip("/") + '_sox_normed/'

    
    print("Computing convolutions and reverberant mfccs...")
    make_convolved_dataset(chunkedNormedMusicDir, normedRirDir,  args.outDir, args.window_size,
                           args.mfcc_bands, args.mfcc_degree, args.chunk_duration_s)


