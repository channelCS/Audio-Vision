# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import LSTM, Merge, Dense, Embedding, Input,Bidirectional
from keras.models import Model
from keras.layers import merge

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
    
    # Make image model
    inpx1=Input(shape=(img_vec_dim,))
    x1=Dense(1024, activation=activation_1)(inpx1)
    x1=Dropout(dropout)(x1)
    image_model = Model([inpx1],x1)
    image_model.summary()
    
    # Make language Model
    inpx0=Input(shape=(max_ques_length,))
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx0)
    x1=LSTM(num_hidden_units_lstm, return_sequences=True)(x0)
    x1=LSTM(num_hidden_units_lstm, return_sequences=True)(x1)
    x2=LSTM(num_hidden_units_lstm, return_sequences=False)(x1)
    x2=Dense(1024,activation=activation_2)(x2)
    x2=Dropout(dropout)(x2)

    # Make embedding_model
    embedding_model = Model([inpx0],x2)
    embedding_model.summary()
    
    # Make combined model
    model = Sequential()
    model.add(Merge([image_model,embedding_model],mode = 'mul'))
    for i in xrange(num_hidden_layers_mlp):
        model.add(Dense(num_hidden_units_mlp))
        model.add(Activation(activation_1))
        model.add(Dropout(dropout))
    model.summary()
    model.add(Dense(nb_classes))
    model.add(Activation(class_activation))

    return model

def visual_lstm(img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,embedding_matrix):
    
    # Make image model
    inpx1=Input(shape=(img_vec_dim,))
    x1=Dense(embedding_matrix.shape[1], activation='tanh')(inpx1)
    x1=Reshape((1,embedding_matrix.shape[1]))(x1)
    image_model = Model([inpx1],x1)
    image_model.summary()
    
    # Make language Model
    inpx0=Input(shape=(max_ques_length,))
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx0)
    x2=Dense(embedding_matrix.shape[1],activation='tanh')(x0)
    x2=Dropout(dropout)(x2)

    # Make embedding_model
    embedding_model = Model([inpx0],x2)
    embedding_model.summary()
    
    # Make combined model
    model = Sequential()
    model.add(Merge([image_model,embedding_model],mode = 'concat', concat_axis=1))
    model.add(LSTM(num_hidden_units_lstm, return_sequences=False, go_backwards=True))
    model.add(Dense(num_hidden_units_mlp))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.summary()
    model.add(Dense(nb_classes))
    model.add(Activation(class_activation))

    return model

def visual_lstm2(img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,embedding_matrix):
    
    # Make image model
    inpx1=Input(shape=(img_vec_dim,))
    x1=Dense(embedding_matrix.shape[1], activation=activation_1)(inpx1)
    x1=Reshape((1,embedding_matrix.shape[1]))(x1)
    image_model = Model([inpx1],x1)
    image_model.summary()
    
    # Make language Model
    inpx0=Input(shape=(max_ques_length,))
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx0)
    x2=Dense(embedding_matrix.shape[1],activation=activation_2)(x0)
    x2=Dropout(dropout)(x2)

    # Make embedding_model
    embedding_model = Model([inpx0],x2)
    embedding_model.summary()
    
    inpx2=Input(shape=(img_vec_dim,))
    x1=Dense(embedding_matrix.shape[1], activation=activation_1)(inpx1)
    x3=Reshape((1,embedding_matrix.shape[1]))(x1)
    image_model2 = Model([inpx2],x3)
    image_model2.summary()
    
    # Make combined model
    model = Sequential()
    model.add(Merge([image_model,embedding_model, image_model2],mode = 'concat', concat_axis=1))
    model.add(Bidirectional(LSTM(num_hidden_units_lstm, return_sequences=False)))
    model.add(Dense(num_hidden_units_mlp))
    model.add(Activation(activation_1))
    model.add(Dropout(dropout))
    
    model.summary()
    model.add(Dense(nb_classes))
    model.add(Activation(class_activation))

    return model
