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

import matplotlib.pyplot as plt

import pandas as pd
from progressbar import progressbar


#hide warnings 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-audioDir', '--audioDir',
        type=str, default='datasets/final_ds/Test/speech/',
        help='Audio directory.'
    )
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, default='datasets/final_ds/Test/RIR_final/',
        help='rir directory'
    )
    
    parser.add_argument(
        '-gpu', '--gpu',
        type=int, default=0, 
        help='GPU used for computation '
    )
    
    parser.add_argument(
        '-audioDuration', '--audioDuration',
        type=int, default=15, 
        help='Length of audio signal to use in seconds '
    )
    
    parser.add_argument(
        '-nbBands', '--nbBands', 
        type = int, default = 6,
        help='number of output bands, 6 or 12 (6 for joint estimation)'
    )
    
    parser.add_argument(
        '-param', '--param',
        type=str, default='t60', 
        help='Parameter to predict among t60,c50,c80,drr,all '
    )
    
    parser.add_argument(
        '-model', '--model',
        type=str, default='CRNN2D_2', 
        help='Network to use for inference '
    )
    
    parser.add_argument(
        '-loadWeights', '--loadWeights',
        type=str, default='thesis_trainings/weights/weights.best.t60_speech_0db.hdf5', 
        help='If you want to specify which weights to use'
    )
    
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    
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

def predict(mfcc_dir,audio_list,true_ac_dir,params):
    df = pd.DataFrame(columns = os.listdir(mfcc_dir),
                     index=audio_list)

    for room in progressbar(df.columns):
        for track in df.index:
            with open(os.path.join(mfcc_dir,room,track),'rb') as f:
                X = pickle.load(f)
            f.close()

            df.loc[track,room] = model.predict(X.reshape((1,params['mfcc_bands'], 
                                                          params['window_size'],
                                                          params['n_channels'])))
            
    return df
#---------------------------------------------------------------------------------
def plot_room_pred(room_name, preds_df, target_df, bands, param,saveplots=True):
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
    if (saveplots):
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/'+room_name+'.png')
        
    plt.show()
#---------------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    #setup GPU
    tfback._get_available_gpus = _get_available_gpus

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    #Organize directories
    rir_test_dir = args.rirDir
    audio_test_dir = args.audioDir
    rir_normed_dir = rir_test_dir.strip('/') + '_sr16000_c_1/'

    mfcc_dir = os.path.join(os.path.split(audio_test_dir.strip('/'))[0],'mfcc_' + os.path.split(audio_test_dir.strip('/'))[1])    
    
    true_values_dir = os.path.join(os.path.split(rir_test_dir.strip('/'))[0],'true_acoustic_values')

    #Prepare dataset
    print(f"preparing dataset")
    
    if os.path.exists(mfcc_dir):
        recompute_conv = input("mfcc directory already exists, do you want to recompute it? y or n : ")
    else:
        os.makedirs(mfcc_dir)
        recompute_conv = 'y'
 
    if recompute_conv == 'y':
        call('python convolute.py -audioDir ' + audio_test_dir + 
             ' -rirDir ' + rir_test_dir + 
             ' -outDir ' + mfcc_dir +
             ' -trim ' + str(args.audioDuration) +
             ' -outFormat mfcc', shell=True)

    #Compute and save true values
    if os.path.exists(true_values_dir):
        recompute_true_vals = input("true value directory already exists, do you want to recompute it? y or n : ")
    else:
        os.makedirs(true_values_dir)    
        recompute_true_vals = 'y'

    if recompute_true_vals == 'y':
        if args.nbBands == 6:
            call('python acoustic_param_ds_maker.py -rirDir ' + rir_normed_dir + 
                 ' -outDir ' + true_values_dir + 
                 ' -params ' + args.param,
                 shell=True)
        elif args.nbBands == 12:
            call('python acoustic_param_ds_maker.py -rirDir ' + rir_normed_dir + 
                 ' -outDir ' + true_values_dir + 
                 ' -params ' + args.param + 
                 ' -bands 500 630 800 1000 1250 1600 2000 2500 3150 4000 5000 6300', 
                 shell=True)

    
    #Define model
    params = {'window_size': args.audioDuration*100 -2,
              'mfcc_bands': 40,
              'n_channels': 1,
              'output_size': args.nbBands}

    model  = CRNN2D((None,params['mfcc_bands'], 
                 params['window_size'],
                 params['n_channels']), 
                params['output_size'])
    
    model.load_weights(args.loadWeights)
    
    #Make predictions
    print("Making predictions")
    audio_list = [os.path.splitext(os.path.basename(file))[0]+'.pkl' for file in glob.glob(audio_test_dir+'*.wav')]
    
    pred_file = 'predictions.csv'
    if os.path.exists(pred_file):
        redo_preds = input('predictions.csv already exists, overwrite it? y or n : ')
        if redo_preds == 'n':
            sys.exit(1)
    
    preds_df = predict(mfcc_dir,audio_list,true_values_dir,params)
    #save predictions
    preds_df.to_csv(pred_file,index=True)
   
    #Load true values
    target_df = pd.DataFrame(columns = next(os.walk(mfcc_dir))[1],
                     index=next(os.walk(true_values_dir))[1])
    
    for param in target_df.index:
        for room in target_df.columns:
            with open(os.path.join(true_values_dir,param,room) + '.pkl','rb') as f:
                true_val = pickle.load(f)
                target_df.loc[param,room] = true_val
            f.close()
            
    #plot
    if args.plot:
        for room in preds_df.columns:
            plot_room_pred(room, preds_df, target_df, bands, param = args.param)
            
    #print error
    res_arr = np.empty(len(preds_df))
    

    for i,col in enumerate(preds_df.columns):
        preds_arr = np.vstack(preds_df[str(col)].values)
        target_arr = np.vstack(len(preds_arr)*[(target_df[str(col)].loc[args.param])])
        res_arr[i] = mean_absolute_error(target_arr,preds_arr)
        
    print(f"mean error = {res_arr.mean()}")
    
    
    