from argparse import ArgumentParser

# Load various imports 
import pandas as pd
import os
import librosa
import glob
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import sys
import pickle

from subprocess import call

from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
from keras.models import load_model
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
import datetime

from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.layers import *
#from tensorflow.keras.models import Sequential
from datetime import datetime

from scipy import stats, signal
import acoustics
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)


from sklearn.metrics import mean_absolute_error
#from keras.callbacks import TensorBoard
from argparse import ArgumentParser

# Custom imports
from utils.models import *
from utils.acoustic_utils import *
from utils.mfcc import compute_norm_mfcc, compute_logspectrogram2

#hide warnings 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-revAudio', '--revAudio',
        type=str, required=True,
        help='reverberant audio file.'
    )
    
    parser.add_argument(
        '-gpu', '--gpu',
        type=int, default=0, 
        help='GPU used for computation '
    )
    
    parser.add_argument(
        '-param', '--param',
        type=str, default='t60', 
        help='Parameter to predict among t60,c50,c80,drr,all '
    )
    
    parser.add_argument(
        '-rir', '--rir',
        type=str, default=None, 
        help='Specify rir to print true output'
    )
    
    parser.add_argument(
        '-model', '--model',
        type=str, default='CRNN2D_largefilters', 
        help='Network to use for inference '
    )
    
    parser.add_argument(
        '-weights', '--weights',
        type=str, required=True, 
        help='Load model weights'
    )
    
    args = parser.parse_args()
    #check args 
    if args.param not in ['t60', 'c50', 'c80', 'drr', 'all']:
        parser.print_help()
        sys.exit(1)
    
    
    return args

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

#---------------------------------------------------------------------------------
def plot_room_pred(room_name, preds_df, target_df, bands, param, plot_dir= None):
    track_names = []
    
    NUM_COLORS = preds_df.shape[0]

    cm = plt.get_cmap('hsv')
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    
    
    plt.suptitle(room_name, fontsize=20)
    plt.xscale('linear')
        
    plt.ylim(bottom=0,top=3)
    for index, track in preds_df.iterrows():
        plt.plot(bands,track[room_name][0])
        track_names.append(index)
        
    plt.plot(bands,target_df.loc[param,room_name], color='red', linewidth = 5)
    #plt.legend((track_names))
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir,room_name + '.png'))
    
    plt.close()
#---------------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    #setup GPU
    tfback._get_available_gpus = _get_available_gpus

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        
    
    #Read audio
    print('Read audio')
    audio_sr, audio_sig = wavfile.read(args.revAudio)
    
    #Compute spectrogram
    print('Compute spectrogram')
    mfcc = compute_logspectrogram2(audio_sig, audio_sr)
    
    #Define model
    print('Initialize model')
    if args.param == 'all':
        #Output is all parameters stacked
        output_size = 4
        output_labels = ['t60','c50','c80','drr']
    else:
        #Output is a frequency depend parameter
        output_size = 6
        output_labels = [125,  250,  500, 1000, 2000, 4000]
        
    model = eval(args.model)((None,mfcc.shape[0],mfcc.shape[1],1),output_size)
    
    #Load weights
    print('Load weights')
    model.load_weights(args.weights)

    #Predict
    print('Make predictions')
    predictions = model.predict(mfcc.reshape((1,mfcc.shape[0],mfcc.shape[1],1))).squeeze()
        
    #Compute true output if rir file is specified:
    if args.rir is not None:
        print('Compute true acoustic parameters')
        #Read rir file
        try:
            rir_sr, rir = wavfile.read(args.rir)
        except:
            print("Error encountered while parsing file: ",args.rir)
            sys.exit(1)
            
        #Specify bandpass filter center frequency for parameter computations:
        bands = np.array([125.,  250.,  500., 1000., 2000., 4000.])
        
        if args.param == 'all':
            t60 = t60_impulse(rir,rir_sr, bands,rt='t30')
            c50 = acoustics.room.clarity(50, rir, rir_sr, bands)
            c80 = acoustics.room.clarity(80, rir, rir_sr, bands)
            drr = drr_impulse(rir, rir_sr, bands)
            
            true_values = np.hstack((t60.mean(),c50.mean(),c80.mean(),drr.mean()))
        elif args.param == 't60':
            true_values = t60_impulse(rir,rir_sr, bands,rt='t30')
        elif args.param == 'c50':
            true_values = acoustics.room.clarity(50, rir, rir_sr, bands)
        elif args.param == 'c80':
            true_values = acoustics.room.clarity(80, rir, rir_sr, bands)
        elif args.param == 'drr':
            true_values = drr_impulse(rir, rir_sr, bands)

            
    print(output_labels)
    print('--- True values ---')
    print(true_values)
    print('--- Estimations ---')
    print(predictions)
    
    print(f'\n Mean absolute error: {mean_absolute_error(true_values,predictions)}')
    