# Load various imports 
import pandas as pd
import os
import librosa
import glob
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

import pickle

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

#from keras.callbacks import TensorBoard
from time import time
from argparse import ArgumentParser

# Custom imports
from utils.models import *

#---------------------------------------------------------------------------------
# Parameters

def parse_args():
    parser = ArgumentParser(description='Training')
    
    parser.add_argument(
        '-xDir', '--xDir',
        type=str, default=None, required=True,
        help='Mfcc data directory.'
    )
    
    parser.add_argument(
        '-yDir', '--yDir',
        type=str, default=None, required=True,
        help='true acoustic values (from rir) directory.'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, default='logs',
        help='output directory where to save logs and weights'
    )
    
    parser.add_argument(
        '-name', '--name',
        type=str, default=None, required = True,
        help='Training label /!\ No space please.'
    )
    
    parser.add_argument(
        '-load_weights', '--load_weights',
        type=str, default=None,
        help='path to weights to load, default None does not load any.'
    )

    parser.add_argument(
        '-n_epochs', '--n_epochs',
        type=int, default=200,
        help='Number of epochs before finish.'
    )
    
    parser.add_argument(
        '-window_size', '--window_size',
        type=int, default=251,
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
        '-n_channels', '--n_channels',
        type=int, default=1,
        help='Numbers of mfccs stacked for input'
    )
    
    parser.add_argument(
        '-output_size', '--output_size',
        type=int, default=12,
        help='output shapes [number of freq bands]'
    )
    
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=64,
        help='Network batch size'
    )
    
    parser.add_argument(
        '-y_param', '--y_param',
        type=str, default='t60',
        help='Output to learn : t60, drr, c50, c80, all'
    )
    
    parser.add_argument(
        '-net', '--net',
        type = str, default = 'CRNN2D',
        help='which neural net to train')
    
    parser.add_argument(
        '-lr', '--learning_rate',
        type = float, default = 0.001,
        help='adam learning rate')
    
    parser.add_argument(
        '-gpu', '--gpu',
        type = int, default = 1,
        help = 'gpu 0 or gpu 1 ')
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

#---------------------------------------------------------------------------------

"""
Input parameters :
dataset_folder = path to where dataset resides
list_IDs = array of ids 
mfcc_degree = 0->mfcc ; 1->mfcc+delta ; 2->mfcc+delta+deltadelta
window_size = number of frames (cols)
mfcc_bands = number of spectrogram frequency buckets (rows)
output_size = number of frequency bands to predict
n_channels = number of parallel input mfcc
batch_size = size of the batch
shuffle 
"""    


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, y_dir, ds_paths,
                 window_size, mfcc_bands,
                 mfcc_degree, y_param,
                 n_channels,
                 output_size,
                 batch_size, 
                 shuffle=True):
        'Initialization'
        self.ds_paths = ds_paths
        self.y_dir = y_dir
        self.window_size = window_size
        self.mfcc_bands = mfcc_bands
        self.y_param = y_param
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_size = output_size
        self.params = ['t60','drr','c50','c80']
        #print(f'output size is {self.output_size}')
        #all combinaisons of music and rirs # format = [music_name, rir_name]
        #self.combinations = np.stack(np.meshgrid(self.music_list,self.rir_list),-1).reshape(-1,2)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ds_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_paths_temp = [self.ds_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_paths_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ds_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        ###### X and y Initialization

        X = np.empty((self.batch_size, self.mfcc_bands, self.window_size, self.n_channels))
       
        if self.output_size == 1:
            y = np.empty(self.batch_size)
        else: 
            y = np.empty((self.batch_size, self.output_size))

        ###### Generate data
        
        #Load mfccs
        for i, path in enumerate(list_paths_temp):
            
            with open(path, "rb") as f:
                X[i] = pickle.load(f).reshape(1, self.mfcc_bands, self.window_size, self.n_channels)
            f.close()
            
            if(np.isnan(np.sum(X[i]))):
                print(f'\nWARNING : Nan value found in {i}')            

            #Load output
            
            room_name = os.path.split(os.path.dirname(path))[1]
            
            if self.y_param == 'all':
                for j,param in enumerate(self.params):
                    with open(self.y_dir + param + '/' + room_name + '.pkl', "rb") as f:
                        buffer = np.concatenate(buffer, pickle.load(f))
                    f.close()
                y[i] = buffer
                
            else:
                with open(self.y_dir + self.y_param + '/' + room_name + '.pkl', "rb") as f:
                    y[i] = pickle.load(f)
                f.close()
                
                if(np.isnan(np.sum(y[i]))):
                    print(f'\nWARNING : Nan value found in {index}')
                    #if i==1: print(f'\ny value = {y[i]}\n')
        return X, y
    
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Parameters
    params = {'window_size': args.window_size,
              'mfcc_bands': args.mfcc_bands,
              'mfcc_degree': args.mfcc_degree,
              'y_param': str(args.y_param),
              'n_channels': args.n_channels,
              'batch_size': args.batch_size,
              'output_size': args.output_size,
              'shuffle': False}
    
    #Setup GPUs environment
    tfback._get_available_gpus = _get_available_gpus

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

    
    print("\n\n-----Parameters-----\n")
    for i in params:
        print(f'{i} : {params[i]}')
    print('\n')
    
    
    print("parsing x file paths...")
    paths_list = glob.glob(args.xDir + '*/*.pkl')
    print('done!')
    
    #Split train/test
    train_ds, test_ds = train_test_split(paths_list,test_size=0.2, random_state=42)

    # Generators
    training_generator = DataGenerator(args.yDir, train_ds, **params)
    validation_generator = DataGenerator(args.yDir, test_ds, **params)

    # Define model
    model = eval(args.net)((None,params['mfcc_bands'], params['window_size'],params['n_channels']), args.output_size)

    #model = CRNN2D((None,40, 645 ,2), 13)
    
    #load previous weights
    if(args.load_weights):
        print("\n------Loading weights------\n")
        model.load_weights(args.load_weights)

    #Setup optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer = opt) 

    #create directory to save weights and logs
    if not os.path.exists(args.outDir + 'logs'):
        os.makedirs(args.outDir + 'logs')
    if not os.path.exists(args.outDir + 'weights'):
        os.makedirs(args.outDir + 'weights')
    
    #Callbacks
    checkpointer = ModelCheckpoint(filepath= args.outDir + 'weights/' + 'weights.best.' + str(args.name) + '.hdf5', verbose=1, save_best_only=True)

    logdir = args.outDir + 'logs/' + str(args.name)# + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=logdir,profile_batch=0,update_freq='batch')
    
    # Train model on dataset
    model.fit(training_generator,
              validation_data=validation_generator,
              epochs = args.n_epochs,
              use_multiprocessing=False,
              callbacks=[checkpointer,tensorboard],
              workers=1)


    evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

