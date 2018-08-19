"""
Created on Sat Apr 08 11:48:18 2018

author: @akshitac8, @adityac8
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose, merge
from keras.layers import BatchNormalization, Lambda,Activation,Concatenate,RepeatVector,Dot,dot
from keras.layers import LSTM, GRU, Reshape, Bidirectional, Permute,TimeDistributed
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Multiply
from keras import optimizers, metrics
from keras import backend as K
import numpy as np

############################# Keras shape #############################
def kr(t,m=None):
    if m is None:
        return t._keras_shape
    else:
        return t._keras_shape[m]

###########################FUNCTIONAL MODELS#############################################
    
########################### BASIC DNN #################################
def dnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 3 Dense layers each followed by a Dropout.
    
    Parameters
    ----------
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
    dropout : float
        default : 0.1
        Dropout used after each Dense Layer.
    act1 : str
        default : relu
        Activation used after 1st layer.
    act2 : str
        default : relu
        Activation used after 2nd layer.
    act3 : str
        default : relu
        Activation used after 3rd layer.
    act4 : str
        default : softmax
        Activation used after 4th layer.
    print_sum : bool
        default : False
        Print summary if the model
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    DNN Model
    """
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    dropout       = kwargs['kwargs'].get('dropout',0.1)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','relu')
    act3          = kwargs['kwargs'].get('act3','relu')
    act4          = kwargs['kwargs'].get('act4','softmax')
    print_sum     = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')
    print("Model DNN")
    print("Activation 1 {} 2 {} 3 {} 4 {}".format(act1,act2,act3,act4))
    print("Neurons {} Dropout {}".format(input_neurons,dropout))
    print("Loss {} Optimizer {} Metrics {}".format(loss,optimizer,metrics))
#    input_dim = dimx * dimy
#    inpx = Input(shape=(input_dim,))
    base_model=seq2seq(dimx,dimy,num_classes,'seq2seq_weights.h5')
    inpx=base_model.get_layer('dense_5').output
    x = Dense(input_neurons, activation=act1)(inpx)
    x = Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act1)(x)
    x = Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(x)
    x = Dropout(dropout)(x)
    score = Dense(num_classes, activation=act3)(x)
    model = Model([base_model.input],score)
    if print_sum:
        model.summary()
#    model.load_weights('seq2seq_weights.h5')
    model.compile(loss=loss,optimizer=optimizer,metrics=[metrics])
    
    return model

########################### BASIC CNN #################################
def cnn(dimx,dimy,num_classes,**kwargs):
    """
    Convolution Neural Network containing 1 Convolution layer followed by a 
    Dense Layer.
    
    Parameters
    ----------
    input_neurons : int
        default : 200
        Number of Neurons for the Dense layer.
    dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after 1st Convolution layer.
    act2 : str
        default : relu
        Activation used after 1st Dense layer.
    act3 : str
        default : softmax
        Activation used after last Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
    nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    CNN Model
    """

    input_neurons  = kwargs['kwargs'].get('input_neurons',200)
    act1           = kwargs['kwargs'].get('act1','relu')
    act2           = kwargs['kwargs'].get('act2','relu')
    act3           = kwargs['kwargs'].get('act3','softmax')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',128)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',2)
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')
    if type(nb_filter) is int:
        nb_filter = [nb_filter] * 2
    if type(pool_size) is int:
        pool_size = [pool_size] * 2
    print("Model CNN")
    print("Activation 1 {} 2 {} 3 {}".format(act1,act2,act3))
    print("Neurons {} Dropout {}".format(input_neurons,dropout))
    print("Kernels {} Size {} Poolsize {}".format(nb_filter,filter_length,pool_size))
    print("Loss {} Optimizer {} Metrics {}".format(loss,optimizer,metrics))
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    
    x = Conv2D(filters=nb_filter[0],
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)
    hx = MaxPooling2D(pool_size=pool_size[0])(x)
    
    x = Conv2D(filters=nb_filter[1],
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)

    hx = MaxPooling2D(pool_size=pool_size[1])(x)
    h = Flatten()(hx)
    wrap = Dense(input_neurons, activation=act2,name='wrap')(h)
    wrap= Dropout(dropout)(wrap)
    score = Dense(num_classes,activation=act3,name='score')(wrap)
    
    model = Model([inpx],score)
    if print_sum:
        model.summary()
    model.load_weights('seq2seq_weights.h5')
    model.compile(loss=loss,optimizer=optimizer,metrics=[metrics])
    
    return model

########################### BASIC RNN #################################
def rnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 1 LSTM layers followed by 3 Dense Layers.
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
    act1 : str
        default : relu
        Activation used after 1st layer.
    act2 : str
        default : relu
        Activation used after 2nd layer.
    act3 : str
        default : relu
        Activation used after 3rd layer.
    act4 : str
        default : softmax
        Activation used after 4th layer.
    print_sum : bool
        default : False
        Print summary if the model
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    RNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',32)
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','relu')
    act3          = kwargs['kwargs'].get('act3','relu')
    act4          = kwargs['kwargs'].get('act4','sigmoid')
    print_sum     = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')

    
#    input_dim=dimx*dimy
#    main_input = Input(shape=(input_dim,), name='main_input')
    base_model=seq2seq(dimx,dimy,num_classes,'seq2seq_weights.h5')
    inpx=base_model.get_layer('dense_5').output
    x = Dense(input_neurons, activation=act1)(inpx)
    a,b=kr(x)
    x=Reshape((1,b))(x)
    x = LSTM(rnn_units)(x)

    # We stack a deep densely-connected network on top
    x = Dense(input_neurons, activation=act1)(x)
    x = Dense(input_neurons, activation=act2)(x)
    x = Dense(input_neurons, activation=act3)(x)
    
    # And finally we add the main logistic regression layer
    main_output = Dense(num_classes, activation=act4, name='main_output')(x)
    model = Model(inputs=base_model.input, outputs=main_output)
    if print_sum:
        model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

    return model

########################### BASIC CRNN #################################
def cnn_rnn(dimx,dimy,num_classes,**kwargs):
    """
    Deep Neural Network containing 1 LSTM layers followed by 3 Dense Layers.
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
    input_neurons : int
        default : 200
        Number of Neurons for each Dense layer.
	dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after Convolution layer.
    act2 : str
        default : tanh
        Activation used after Recurrent layer.
    act3 : str
        default : softmax
        Activation used after Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
	nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    RNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',32)
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','tanh')
    act3          = kwargs['kwargs'].get('act3','softmax')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')


    main_input = Input(shape=(1,dimx,dimy))
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1)(main_input)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    wrap= Dropout(dropout)(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x)
    x = LSTM(rnn_units,activation=act2)(x)
    wrap= Dropout(dropout)(x)
    x = Dense(input_neurons, activation=act3)(wrap)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

    return model



############################# BASIC CBRNN #############################
def cbrnn(dimx,dimy,num_classes,**kwargs):
    """
    CNN with biderectional lstm
    
    Parameters
    ----------
    rnn_units : int
        default : 32
        Number of Units for LSTM layer.
	dropout : float
        default : 0.1
        Dropout used after the Dense Layer.
    act1 : str
        default : relu
        Activation used after 4 Convolution layers.
    act2 : str
        default : sigmoid
        Activation used after Recurrent layer.
    act3 : str
        default : sigmoid
        Activation used after Dense layer.
    print_sum : bool
        default : False
        Print summary if the model
	nb_filter : int
        default : 100
        Number of kernels
    filter_length : int, tuple
        default : 3
        Size of kernels
    pool_size : int, tuple
        default : (2,2)
        Pooling size.
    loss
        default : categorical_crossentropy
        Loss used
    optimizer
        default : adam
        Optimizer used
    metrics
        default : accuracy
        Metrics used.
        
    Returns
    -------
    CBRNN Model
    """
    rnn_units     = kwargs['kwargs'].get('rnn_units',64)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','sigmoid')
    act3          = kwargs['kwargs'].get('act3','sigmoid')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',5)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')


#    print("Functional CBRNN")
#    print("Activation 1 {} 2 {} 3 {}".format(act1,act2,act3))
#    print("Dropout {}".format(dropout))
#    print("Kernels {} Size {} Poolsize {}".format(nb_filter,filter_length,pool_size))
#    print("Loss {} Optimizer {} Metrics {}".format(loss,optimizer,metrics))

    main_input = Input(shape=(1,dimx,dimy))
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(main_input)
    #x1=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
    #x2=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=pool_size)(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
    #x3=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(2,2))(x)
#    wrap= Dropout(dropout)(hx)
    
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
#    x4=BatchNormalization()(x)
    hx = MaxPooling2D(pool_size=(1,1))(x)
    wrap= Dropout(dropout)(x)
    
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
#    x = Reshape((c*d,b))(x) 
    
#    w = Bidirectional(LSTM(rnn_units,activation=act2,return_sequences=False))(x)
    rnnout = Bidirectional(LSTM(rnn_units, activation=act2, return_sequences=True))(x)
    rnnout_gate = Bidirectional(LSTM(rnn_units, activation=act3, return_sequences=False))(x)
    w = Multiply()([rnnout, rnnout_gate])
    wrap= Dropout(dropout)(w)
    wrap=Flatten()(wrap)
    main_output = Dense(num_classes, activation=act3, name='main_output')(wrap)
    model = Model(inputs=main_input, outputs=main_output)
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])
    
    return model

############################ Multi CNN : Ensemble model combining different features ################################
def multi_cnn(dimx,dimy,num_classes,**kwargs):
    """
    This model is used to combine same or complementary features through a mini ensemble convolution model
    based on their properties.
    """
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    act1          = kwargs['kwargs'].get('act1','relu')
    act2          = kwargs['kwargs'].get('act2','tanh')
    act3          = kwargs['kwargs'].get('act3','softmax')
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    nb_filter      = kwargs['kwargs'].get('nb_filter',100)
    filter_length  = kwargs['kwargs'].get('filter_length',3)
    pool_size      = kwargs['kwargs'].get('pool_size',(2,2))
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','categorical_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','accuracy')

    inps,outs=[],[]
    for i in range(len(dimy)):
        inpx = Input(shape=(1,dimx,dimy[i]))
        inps.append(inpx)
        x = Conv2D(filters=nb_filter,
                   kernel_size=filter_length,
                   data_format='channels_first',
                   padding='same',
                   activation=act1)(inpx)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x= Dropout(dropout)(x)
        h = Flatten()(x)
        outs.append(h)

    combine = merge(mode='concat')(outs) 
    # And finally we add the main logistic regression layer    
    wrap = Dense(input_neurons, activation=act2,name='wrap')(combine)
    main_output = Dense(num_classes,activation=act3,name='score')(wrap)
    
    model = Model(inputs=inps,outputs=main_output)
    if print_sum:
        model.summary()
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

    return model

############################ Transpose CNN ################################
def transpose_cnn(dimx,dimy,num_classes,**kwargs):
    """
    The first section of the neural network contains conv layers.
    The deconv layer after conv layer maintains the same shape.
    The last layer will be a conv layer to calculate class wise score.
    Emphasis is given to check the size parameter for model.
    This is used for acoustic event detection.
    """
    act1          = kwargs['kwargs'].get('act1','tanh')
    act2          = kwargs['kwargs'].get('act2','tanh')
    act3          = kwargs['kwargs'].get('act3','sigmoid')
    nb_filter      = kwargs['kwargs'].get('nb_filter',128)
    pool_size      = kwargs['kwargs'].get('pool_size',(1,2))
    dropout        = kwargs['kwargs'].get('dropout',0.1)
    print_sum      = kwargs['kwargs'].get('print_sum',False)

    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','mse')
    if type(nb_filter) is int:
        nb_filter = [nb_filter] * 2
    inpx = Input(shape=(1,dimx,dimy),name='inpx')
    x = Conv2D(filters=nb_filter[0],
               kernel_size=5,
               data_format='channels_first',
               padding='same',
               activation=act1)(inpx)

    hx = MaxPooling2D(pool_size=pool_size)(x)
    #hx = ZeroPadding2D(padding=(2, 1))(hx)
    hx = Conv2D(filters=nb_filter[1],
               kernel_size=3,
               data_format='channels_first',
               padding='same',
               activation=act1)(hx)
   
   
    x=Conv2DTranspose(filters=nb_filter[1], kernel_size=3,padding='same', data_format='channels_first',activation=act2)(hx)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    x=Conv2DTranspose(filters=nb_filter[0], kernel_size=5,padding='same', data_format='channels_first',activation=act2)(hx)
    hx = MaxPooling2D(pool_size=pool_size)(x)
    # Don't use softmax in last layer
    score=Conv2D(filters=num_classes, kernel_size=(1,1),padding='same', data_format='channels_first',activation=act3)(hx)
    # Check for compiling 
#    wrap= Dropout(dropout)(score)
    
    score=GlobalAveragePooling2D(data_format='channels_first')(score)
    kr(score)
    
    model = Model(inputs=[inpx], outputs=[score])
    if print_sum:
        model.summary()
    model.compile(loss=loss,
			  optimizer=optimizer,
			  metrics=[metrics])

    return model

##################### Sequence2Sequence Model ############################
    
def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

def seq2seq(dimx,dimy,num_classes,weights_path=None,**kwargs):
    
    
    # input image dimensions
#    img_rows, img_cols, img_chns = 10, 40, 1
    # number of convolutional filters to use
    filters = 64
    # convolution kernel size
    num_conv = 3
    img_chns = 1
    img_rows = 10
    img_cols = 40
    
#    original_img_size = (img_chns, img_rows, img_cols)
    latent_dim = 2
    intermediate_dim = 128
    batch_size=128
    
    
    x = Input(shape=(1,dimx,dimy))
    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * 5 * 20, activation='relu')
    
    output_shape = (batch_size, filters, 5, 20)
    
    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    output_shape = (batch_size, filters, 11, 41)
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')
    
    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
    
    # instantiate VAE model
    vae = Model(x, x_decoded_mean_squash)
    
    # define the loss function
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
        K.flatten(x),
        K.flatten(x_decoded_mean_squash))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adam')
    
    # load the data
    # Recurrent sequence to sequence learning auto encoders for audio classification task
    
    
    print("seq2seq_lstm")
    
#    ## encoder
##    input_dim=dimx*dimy
#    x = Input(shape=(dimx,dimy))
#    # Encoder
#    encoder=LSTM(64,return_state=True,return_sequences=True)
#    encoder_outputs, forward_h, forward_c= encoder(x)
##    state_h = Concatenate(axis=1)([forward_h, forward_c])
##    state_c = Concatenate(axis=1)([forward_c, backward_c])
#    encoder_states = [forward_h,forward_c]
#    #hidden_1 = Dense(128, activation='relu')(encoder_outputs)
#    #h = Dense(64, activation='relu')(hidden_1)
#    
#    # Decoder
#    y = Input(shape=(dimx,dimy))
#    decoder =LSTM(64, return_sequences=True, return_state=True)
#    decoder_outputs, _, _ = decoder(y,
#                                         initial_state=encoder_states)
#    #hidden_2 = Dense(128, activation='relu')(decoder_outputs)
##    r = Dense(128, activation='relu')(decoder_outputs)
#    r = TimeDistributed(Dense(64, activation='tanh', name="Dense_tanh"))(decoder_outputs)
#    r = TimeDistributed(Dense(dimy, activation="softmax"))(r)
#
#    
##    attention = dot((2,2))([decoder, encoder])
##    attention = dot([Dense(dimx)(encoder_outputs),Dense(dimy)(decoder_outputs)],axes=1)
##    attention = Activation('softmax')(attention)
##    
##    context = Dot((2,1))([attention, encoder])
##    decoder_combined_context = Concatenate(axis=-1)([context, decoder])
##    
##    # Has another weight + tanh layer as described in equation (5) of the paper
##    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
##    output = TimeDistributed(Dense(dimy, activation="softmax"))(output)
#    
#    
#    model = Model([x,y], r)
#    if weights_path:
#        model.load_weights(weights_path)
#    model.summary()
#    model.compile(optimizer='adadelta', loss='mse',metrics=['mse'])
    
    
    return vae


####################### ATTENTION MODEL ACRNN ##################################


def ACRNN(dimx,dimy,num_classes,**kwargs):
    act1          = kwargs['kwargs'].get('act1','tanh')
    nb_filter     = kwargs['kwargs'].get('nb_filter',72)
    filter_length = kwargs['kwargs'].get('filter_length',4)
    
    act2          = kwargs['kwargs'].get('act2','linear')
    rnn_units     = kwargs['kwargs'].get('rnn_units',[20,20])    
    dropout       = kwargs['kwargs'].get('dropout',[0.1,0.2])
    
    act3          = kwargs['kwargs'].get('act3','softmax')
    print_sum      = kwargs['kwargs'].get('print_sum',False)
    
    loss          = kwargs['kwargs'].get('loss','binary_crossentropy')
    optimizer     = kwargs['kwargs'].get('optimizer','adam')
    metrics       = kwargs['kwargs'].get('metrics','mse')
    
    #input shape
    main_input = Input(shape=(1,dimx,dimy))
    
    #CNN
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(main_input)
    hx = MaxPooling2D(pool_size=(1,2))(x)
    x = Conv2D(filters=nb_filter,
               kernel_size=filter_length,
               data_format='channels_first',
               padding='same',
               activation=act1,use_bias=False)(hx)
    hx = MaxPooling2D(pool_size=(1,2))(x)
    wrap= Dropout(dropout[0])(hx)
    x = Permute((2,1,3))(wrap)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x) 
    
    #RNN LAYERS
    rnnout = Bidirectional(GRU(rnn_units[0],activation=act2,  return_sequences=True),merge_mode='concat')(x)
    rnnout_1      = Bidirectional(GRU(rnn_units[1],activation='sigmoid', return_sequences=True),merge_mode='concat')(rnnout)
    w = Multiply()([rnnout, rnnout_1])
    
    
    #Attention starts
    hidden_size = int(w._keras_shape[1])
    a = Permute((2, 1))(w)
    a = Reshape((hidden_size, a._keras_shape[1]))(a) 
    a = TimeDistributed(Dense( a._keras_shape[1], activation='softmax',use_bias=False))(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(dimy)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_mul = merge([w,a_probs], name='attention_mul', mode='mul')
    
    attention_mul = GlobalMaxPooling1D()(attention_mul)
    attention_mul = Dropout(dropout[1])(attention_mul)
    
    # compile Model
    main_output = Dense(num_classes, activation=act3)(attention_mul)
    mymodel = Model([main_input], main_output)
    if print_sum:
        mymodel.summary()
    mymodel.compile(loss=loss,
			  optimizer=optimizer,
			  metrics=[metrics])

    return mymodel



########################################### DYNAMIC MODELS ###########################################
"""
Dynamic Models can be accessed by 

"""
########################### DYNAMIC DNN #################################
def dnn_dynamic(num_classes,input_dim,acts,**kwargs):
    input_neurons = kwargs['kwargs'].get('input_neurons',200)
    drops         = kwargs['kwargs'].get('drops',[])
    dnn_layers    = kwargs['kwargs'].get('dnn_layers',1)
    last_act      = kwargs['kwargs'].get('last_act','softmax')
    end_dense = kwargs['kwargs'].get('end_dense',{})

    
    if not np.all([len(acts)==dnn_layers]):
        print("Layers Mismatch")
        return False
    x = Input(shape=(input_dim,),name='inpx')
    inpx = x
    for i in range(dnn_layers):
        x = Dense(input_neurons,activation=acts[i])(inpx)
        if drops != []:
            x = Dropout(drops[i])(x)

    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    score = Dense(num_classes,activation=last_act,name='score')(x)
    
    model = Model(inpx,score)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

########################### DYNAMIC CNN #################################
def cnn_dynamic(num_classes,dimx,dimy,acts,**kwargs):
    cnn_layers = kwargs['kwargs'].get('cnn_layers',1)
    nb_filter     = kwargs['kwargs'].get('nb_filter',[])
    filter_length = kwargs['kwargs'].get('filter_length',[])

    pools     = kwargs['kwargs'].get('pools',[])
    drops     = kwargs['kwargs'].get('drops',[])
    bn        = kwargs['kwargs'].get('batch_norm',False)
    end_dense = kwargs['kwargs'].get('end_dense',{})
    last_act  = kwargs['kwargs'].get('last_act','softmax')

    if not np.all([len(acts)==cnn_layers,len(nb_filter)==cnn_layers,len(filter_length)==cnn_layers]):
        raise Exception("Layers Mismatch")
    x = Input(shape=(1,dimx,dimy),name='inpx')
    inpx = x
    for i in range(cnn_layers):
        x = Conv2D(filters=nb_filter[i],
                   kernel_size=filter_length[i],
                   data_format='channels_first',
                   padding='same',
                   activation=acts[i])(x)
        if bn:
            x=BatchNormalization()(x)
        if pools != []:
            if pools[i][0]=='max':
                x = MaxPooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='avg':
                x = AveragePooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='globmax':
                x = GlobalMaxPooling2D()(x)
            elif pools[i][0]=='globavg':
                x = GlobalAveragePooling2D()(x)
        if drops != []:
            x = Dropout(drops[i])(x)

    if pools[-1][0]=='max' or pools[-1][0]=='avg':
        x = Flatten()(x)
    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    score = Dense(num_classes,activation=last_act,name='score')(x)
    
    model = Model(inpx,score)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    return model

############################# DYNAMIC CBRNN #############################
def cbrnn_dynamic(num_classes,dimx,dimy,acts,**kwargs):
    """
    """
    pools     = kwargs['kwargs'].get('pools',[])
    drops     = kwargs['kwargs'].get('drops',[])
    bn        = kwargs['kwargs'].get('batch_norm',False)
    end_dense = kwargs['kwargs'].get('end_dense',{})
    last_act  = kwargs['kwargs'].get('last_act','softmax')
    
    cnn_layers = kwargs['kwargs'].get('cnn_layers',1)
    rnn_layers = kwargs['kwargs'].get('rnn_layers',1)
    rnn_type   = kwargs['kwargs'].get('rnn_type','LSTM')
    rnn_units  = kwargs['kwargs'].get('rnn_units',[])
    nb_filter     = kwargs['kwargs'].get('nb_filter',[])
    filter_length = kwargs['kwargs'].get('filter_length',[])
    #CNN with biderectional lstm
    print("CBRNN")
    if not np.all([len(acts)==cnn_layers,len(nb_filter)==cnn_layers,len(filter_length)==cnn_layers]):
        print("Layers Mismatch")
        return False
    x = Input(shape=(1,dimx,dimy),name='inpx')
    inpx = x
    for i in range(cnn_layers):
        x = Conv2D(filters=nb_filter[i],
                   kernel_size=filter_length[i],
                   data_format='channels_first',
                   padding='same',
                   activation=acts[i])(x)
        if bn:
            x=BatchNormalization()(x)
        if pools != []:
            if pools[i][0]=='max':
                x = MaxPooling2D(pool_size=pools[i][1])(x)
            elif pools[i][0]=='avg':
                x = AveragePooling2D(pool_size=pools[i][1])(x)
        if drops != []:
            x = Dropout(drops[i])(x)
    x = Permute((2,1,3))(x)
    a,b,c,d= kr(x)
    x = Reshape((b*d,c))(x)

    for i in range(rnn_layers):
        #Only last layer can have return_sequences as False
        r = False if i == rnn_layers-1 else True
        if rnn_type=='LSTM':
            x = LSTM(rnn_units[i],return_sequences=r)(x)
        elif rnn_type=='GRU':
            x = Bidirectional(GRU(rnn_units[i],return_sequences=r))(x)
        elif rnn_type=='bdLSTM':
            x = Bidirectional(LSTM(rnn_units[i],return_sequences=r))(x)
        elif rnn_type=='bdGRU':
            x = Bidirectional(GRU(rnn_units[i],return_sequences=r))(x)
    
    x= Dropout(0.1)(x)
    if end_dense != {}:
        x = Dense(end_dense['input_neurons'], activation=end_dense['activation'],name='wrap')(x)
        try:
            x = Dropout(end_dense['dropout'])(x)
        except:
            pass
    main_output = Dense(num_classes, activation=last_act, name='main_output')(x)
    model = Model(inputs=inpx, outputs=main_output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model
