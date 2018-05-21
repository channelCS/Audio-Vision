from get_data import get_train_data, get_test_data, anat_data, get_val_data
from keras.utils import to_categorical
import os, h5py, json, numpy as np
#from models import deeper_lstm
from my_models import deeper_lstm

root_path = 'D:/workspace/aditya_akshita/vqa/VQA_Keras/data/'
val_file       =root_path+'annotations/mscoco_val2014_annotations.json'
input_json     = root_path+'data_prepro.json'
input_img_h5   = root_path+'data_img.h5'
input_ques_h5  = root_path+'data_prepro.h5'
word_emb_dim          = 300
embedding_matrix_filename = root_path+'embeddings_%s.h5'%word_emb_dim
glove_path                = root_path+'glove.6B.300d.txt'
img_norm       = 1
nb_classes = 1000
optimizer  = 'sgd'
batch_size = 300
epochs     = 4
activation_1          = 'tanh'
activation_2          = 'relu'
dropout               = 0.5
vocabulary_size       = 12603
num_hidden_units_lstm = 512
max_ques_length       = 26
num_hidden_layers_mlp = 3
num_hidden_units_mlp  = 512
class_activation      = 'softmax'
img_vec_dim           = 4096

def get_metadata():
    meta_data = json.load(open(input_json, 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

def prepare_embeddings(num_words, embedding_dim, metadata, embedding_matrix_filename, glove_path):
    if os.path.exists(embedding_matrix_filename):
        with h5py.File(embedding_matrix_filename) as f:
            return np.array(f['embedding_matrix'])

    print "Embedding Data..."  
    embeddings_index = {}
    with open(glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, embedding_dim))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
   
    with h5py.File(embedding_matrix_filename, 'w') as f:
        f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix


"""
dataset, train_img_feature, train_data             = get_train_data(input_json, input_img_h5, input_ques_h5, img_vec_dim, img_norm)
dataset, test_img_feature,  test_data, val_answers = get_test_data( input_json, input_img_h5, input_ques_h5, img_vec_dim, ans_file, img_norm)

train_X = [train_data[u'question'], train_img_feature]
train_Y = to_categorical(train_data[u'answers'], nb_classes)

test_X = [test_data[u'question'], test_img_feature]
test_Y = to_categorical(val_answers, nb_classes)
"""
train_X, train_Y= anat_data(input_img_h5, input_ques_h5)
metadata=get_metadata()

embedding_matrix = prepare_embeddings(vocabulary_size, word_emb_dim, metadata, embedding_matrix_filename, glove_path)

model = deeper_lstm(img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,embedding_matrix)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary() # prints model layers with weights

test_X,test_Y=get_val_data(input_img_h5, input_ques_h5,metadata,val_file)
#model.fit(train_X, train_Y, batch_size = batch_size, epochs=epochs, validation_data=(test_X, test_Y),verbose=1)
model.fit(train_X, train_Y, batch_size = batch_size, epochs=epochs, verbose=1)
