# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import LSTM, Merge, Dense, Embedding, Input,Bidirectional
from keras.models import Model
from keras.layers import merge

def san_atten(common_word_emb_dim,img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,embedding_matrix,filter_sizes,num_attention_layers):
    
    # Image model
    inpx1=Input(shape=(img_vec_dim,))
    x1=Dense(embedding_matrix.shape[1], activation='tanh')(inpx1)
    x1=Reshape((1,embedding_matrix.shape[1]))(x1)
    x2=Dropout(dropout)(x1)
    image_model = Model([inpx1],x2)
    image_model.summary()
    
    # [1] Recurrent (LSTM) question Model
    inpx0=Input(shape=(max_ques_length,))
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx0)
    x2=Dense(embedding_matrix.shape[1],activation='tanh')(x0)
    x2=Dropout(dropout)(x2)
    
    question_model = Model([inpx0],x2)
    question_model.summary()
    
    # [2] CNN question model
    inpx_0=Input(shape=(max_ques_length,))
    
    x0=Embedding(vocabulary_size, word_emb_dim, weights=[embedding_matrix], trainable=False)(inpx_0)
    
    conv_0 = Conv2D(100, kernel_size=(filter_sizes[0], word_emb_dim), padding='valid', kernel_initializer='normal', activation='relu')(x0)
    conv_1 = Conv2D(100, kernel_size=(filter_sizes[1], word_emb_dim), padding='valid', kernel_initializer='normal', activation='relu')(x0)
    conv_2 = Conv2D(100, kernel_size=(filter_sizes[2], word_emb_dim), padding='valid', kernel_initializer='normal', activation='relu')(x0)
    
    maxpool_0 = MaxPool2D(pool_size=(max_ques_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_ques_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_ques_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    output = Dense(embedding_matrix.shape[1],activation='tanh')(dropout)
    print (output.shape)
    # this creates a model that includes
    model = Model(inputs=[inpx_0], outputs=output)
    model.summary()
        
        
    # Stacked Attention Model
    model = Sequential()
    model.add(Merge([image_model,question_model],mode = 'concat', concat_axis=1))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Reshape((1,embedding_matrix.shape[1])))
    model.add(Activation('softmax'))
    model.add(Merge([image_model,question_model],mode = 'mul'))
    