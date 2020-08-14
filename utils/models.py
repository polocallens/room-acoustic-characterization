
from keras.layers import Permute
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.recurrent import GRU, LSTM
from tensorflow.keras.utils import Sequence
#from tensorflow import keras

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential


def CRNN2D(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------


def CRNN2D_2(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model
#---------------------------------------------------------------------------------

def CRNN2D_classifier(X_shape, nb_classes):
    '''
    Model used for evaluation in paper. Inspired by K. Choi model in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    '''

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


#---------------------------------------------------------------------------------
def CNN1(X_shape, nb_classes):

    filter_size = 2

    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape= (X_shape[1], X_shape[2], X_shape[3]), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(nb_classes, activation='relu'))
    
    return model

#---------------------------------------------------------------------------------

def CRNN2D_wo_batchnorm(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    #model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def CRNN2D_largefilters(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [16, 32, 32, 32]  # filter sizes
    kernel_size = (15, 15)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def CRNN2D_1111filters(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [16, 32, 32, 32]  # filter sizes
    kernel_size = (11, 11)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def CRNN2D_99filters(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [16, 32, 32, 32]  # filter sizes
    kernel_size = (9, 9)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model


#---------------------------------------------------------------------------------

def CRNN2D_nopooling(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [16, 32, 32, 32]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    #pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),(4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    #model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        #model.add(MaxPooling2D(pool_size=pool_size[layer+1],strides=pool_size[layer+1],data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model



#---------------------------------------------------------------------------------

def CRNN2D_temporalfilters(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [16, 32, 32, 32]  # filter sizes
    kernel_size = (10, 20)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def CRNN2D_giantfilters(X_shape, nb_classes):

    nb_layers = 4  # number of convolutional layers
    nb_filters = [8, 16, 16, 16]  # filter sizes
    kernel_size = (30, 30)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def CNN(X_shape, nb_classes):

    nb_layers = 6  # number of convolutional layers
    nb_filters = [8, 8, 16, 16,32,32]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 3), (2, 3), (2, 3), (2, 3),
                 (2, 3),(2,3)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
   
    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0],))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1],
                              data_format="channels_first"))  # Max pooling
        model.add(Dropout(0.1))

        
    model.add(Flatten())
    # Output layer
    model.add(Dense(128,activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation=activation))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model

#---------------------------------------------------------------------------------

def MLP(X_shape, nb_classes):

    activation = 'elu'  # activation function to use after each layer
    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    
    
    model = Sequential()
    
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128,activation=activation))
    model.add(Dense(128,activation=activation))
    model.add(Dense(128,activation=activation))
    model.add(Dense(128,activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(128,activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(128,activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation=activation))
    model.add(Dense(nb_classes))
    #model.add(Activation("relu"))
    return model