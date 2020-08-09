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
        '-xDir', '--xDir',
        type=str, required=True,
        help='Audio directory.'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, required=True,
        help='Where to store plots and predictions'
    )
    
    parser.add_argument(
        '-targetDir', '--targetDir',
        type=str, default=None,
        help='True values directory.'
    )
    
    parser.add_argument(
        '-gpu', '--gpu',
        type=int, default=0, 
        help='GPU used for computation '
    )
    
    parser.add_argument(
        '-inputWidth', '--inputWidth',
        type=int, default=798, 
        help='Width of spectrogram input '
    )
    
    parser.add_argument(
        '-inputHeight', '--inputHeight',
        type=int, default=15, 
        help='Length of audio signal to use in seconds '
    )
    
    parser.add_argument(
        '-outputSize', '--outputSize', 
        type = int, default = 6,
        help='number of output bands, 6 or 4 (4 for joint estimation)'
    )
    
    parser.add_argument(
        '-param', '--param',
        type=str, default='t60', 
        help='Parameter to predict among t60,c50,c80,drr,all '
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

def predict(mfcc_dir,audio_list,model_params):
    df = pd.DataFrame(columns = os.listdir(mfcc_dir),
                     index=audio_list)

    for room in progressbar(df.columns):
        for track in df.index:
            with open(os.path.join(mfcc_dir,room,track),'rb') as f:
                X = pickle.load(f)
            f.close()

            df.loc[track,room] = model.predict(X.reshape((1,model_params['mfcc_bands'], 
                                                          model_params['window_size'],
                                                          model_params['n_channels'])))
            
    return df

#---------------------------------------------------------------------------------

def predict_from_dir(mfcc_dir,model_params):
    
    rir_list = os.listdir(mfcc_dir)
    audio_list = os.listdir(os.path.join(mfcc_dir,rir_list[0]))
    
    df = pd.DataFrame(columns = rir_list,
                     index=audio_list)

    for room in progressbar(df.columns):
        for track in df.index:
            with open(os.path.join(mfcc_dir,room,track),'rb') as f:
                X = pickle.load(f)
            f.close()

            df.loc[track,room] = model.predict(X.reshape((1,model_params['mfcc_bands'], 
                                                          model_params['window_size'],
                                                          model_params['n_channels'])))
            
    return df
#---------------------------------------------------------------------------------
def plot_room_pred(room_name, preds_df, target_df, bands, param, plot_dir= None):
    track_names = []
    
    NUM_COLORS = preds_df.shape[0]

    cm = plt.get_cmap('Pastel1')
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
    
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)
        
    #Define model
    model_params = {'window_size': args.inputWidth,
              'mfcc_bands': args.inputHeight,
              'n_channels': 1,
              'output_size': args.outputSize}
    
    bands = [125.,  250.,  500., 1000., 2000., 4000.]

    model  = eval(args.model)((None, args.inputHeight, args.inputWidth,1), 
                              args.outputSize)
    
    model.load_weights(args.weights)
    
    #Make predictions
    
    print("Make predictions")    
    
    if os.path.exists(os.path.join(args.outDir,'predictions.csv')):
        redo_preds = input('predictions.csv already exists, overwrite it? y or n : ')
        if redo_preds == 'n':
            sys.exit(1)
    
    preds_df = predict_from_dir(args.xDir,model_params)
    
    #save predictions
    pred_file = os.path.join(args.outDir,'predictions.csv')
    print(f"Save predictions to {pred_file}")
    preds_df.to_csv(pred_file,index=True)
   
    if args.targetDir is not None:
                
        target_df = pd.DataFrame(columns = os.listdir(args.xDir),
                         index= os.listdir(args.targetDir))

        #Load true values
        for param in target_df.index:
            for room in target_df.columns:
                with open(os.path.join(args.targetDir,param,room) + '.pkl','rb') as f:
                    true_val = pickle.load(f)
                    target_df.loc[param,room] = true_val
                f.close()

        #plot
        plot_dir = os.path.join(args.outDir,'pred_plot')
        print(f"saving plots to {plot_dir}")
        for room in preds_df.columns:
            plot_room_pred(room, preds_df, target_df, bands, param = args.param, plot_dir=plot_dir)

        #print error
        res_arr = np.empty(len(preds_df))

        for i,col in enumerate(preds_df.columns):
            preds_arr = np.vstack(preds_df[str(col)].values)
            target_arr = np.vstack(len(preds_arr)*[(target_df[str(col)].loc[args.param])])
            res_arr[i] = mean_absolute_error(target_arr,preds_arr)

        print(f"mean absolute error = {res_arr.mean()}")

        with open(os.path.join(args.outDir,'errors.txt'),'w') as f:
            f.write(f"mean absolute error = {res_arr.mean()}")
            
    