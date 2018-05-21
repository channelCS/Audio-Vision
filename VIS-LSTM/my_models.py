from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense, Embedding, Input
#import embedding
from keras.models import Model
import h5py
import numpy as np

def kr(t,m=None):
    if m is None:
        return t._keras_shape
    else:
        return t._keras_shape[m]


def basic_mlp(img_vec_dim, vocabulary_size, word_emb_dim,
              max_ques_length, num_hidden_units_lstm, 
              num_hidden_layers_mlp, num_hidden_units_mlp,
              dropout, nb_classes, class_activation):
    # Image model
    model_image = Sequential()
    model_image.add(Reshape((img_vec_dim,), input_shape=(img_vec_dim,)))

    # Language Model
    model_language = Sequential()
    model_language.add(Embedding(vocabulary_size, word_emb_dim, input_length=max_ques_length))
    model_language.add(LSTM(num_hidden_units_lstm, return_sequences=True, input_shape=(max_ques_length, word_emb_dim)))
    model_language.add(LSTM(num_hidden_units_lstm, return_sequences=True))
    model_language.add(LSTM(num_hidden_units_lstm, return_sequences=False))

    # combined model
    model = Sequential()
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))


    for i in xrange(num_hidden_layers_mlp):
        model.add(Dense(num_hidden_units_mlp))
        model.add(Dropout(dropout))

    model.add(Dense(nb_classes))
    model.add(Activation(class_activation))

    return model

def deeper_lstm(img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,embedding_matrix):
    # Image model
    inpx1=Input(shape=(img_vec_dim,))
    x1=Dense(1024, activation=activation_1)(inpx1)
    #x1=Reshape((1024,embedding_matrix.shape[1]))(x1)
    x1=Dropout(dropout)(x1)
     ###Make image_model
    image_model = Model([inpx1],x1)
    image_model.summary()
    


    # Language Model
    inpx0=Input(shape=(max_ques_length,))
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx0)
    x1=LSTM(num_hidden_units_lstm, return_sequences=True)(x0)
    x1=LSTM(num_hidden_units_lstm, return_sequences=True)(x1)
    x2=LSTM(num_hidden_units_lstm, return_sequences=False)(x1)
    x2=Dense(1024,activation=activation_2)(x2)
    x2=Dropout(dropout)(x2)
    ###Make embedding_model
    embedding_model = Model([inpx0],x2)
    embedding_model.summary()
    
    # combined model
    model = Sequential()
    model.add(Merge([image_model,embedding_model],mode = 'mul'))
    model.summary()
    # for _ in xrange(number_of_dense_layers):
    for i in xrange(num_hidden_layers_mlp):
        model.add(Dense(num_hidden_units_mlp))
        model.add(Activation(activation_1))
        model.add(Dropout(dropout))

    model.add(Dense(nb_classes))
    model.add(Activation(class_activation))


    return model


def vis_lstm():
    embedding_matrix = load(embedding_matrix_filename)
    inpx0=Input(shape=(embedding_matrix.shape[0],embedding_matrix.shape[1]))
    x0=Embedding(vocabulary_size, word_emb_dim,weights = [embedding_matrix],trainable = False)(inpx0)
    ###Make embedding_model
    embedding_model = Model([inpx0],x0)
    
    inpx1=Input(shape=(4096,))
    x1=Dense(embedding_matrix.shape[1],activation='linear')(inpx1)
    x1=Reshape((1,embedding_matrix.shape[1]))(x1)
    ###Make image_model
    image_model = Model([inpx1],x1)

    inpx2=Merge([image_model,embedding_model],mode = 'concat',concat_axis = 1)
    x2=LSTM(1001)(inpx2)
    x2=Dropout(0.5)(x2)
    x2=Dense(1001,activation='softmax')(x2)
    ###Make main_model
    main_model = Model([inpx2],x2)
    
    return main_model

def vis_lstm_2():
    embedding_matrix = embedding.load()
    inpx0=Input(shape=(embedding_matrix.shape[0],embedding_matrix.shape[1]))
    x0=Embedding(weights = [embedding_matrix],trainable = False)(inpx0)
    ###Make embedding_model
    embedding_model = Model([inpx0],x0)

    inpx1=Input(shape=(4096,))
    x1=Dense(embedding_matrix.shape[1],activation='linear')(inpx1)
    x1=Reshape((1,embedding_matrix.shape[1]))(x1)
    ###Make image_model_1
    image_model_1 = Model([inpx1],x1)

    inpx2=Input(shape=(4096,))
    x2=Dense(embedding_matrix.shape[1],activation='linear')(inpx2)
    x2=Reshape((1,embedding_matrix.shape[1]))(x2)
    ###Make image_model_2
    image_model_2 = Model([inpx2],x2)

    inpx3=Merge([image_model_1,embedding_model,image_model_2],mode = 'concat',concat_axis = 1)
    x3=LSTM(1001)(inpx3)
    x3=Dropout(0.5)(x3)
    x3=Dense(1001,activation='softmax')(x3)
    ###Make main_model
    main_model = Model([inpx3],x3)

    
    return main_model

