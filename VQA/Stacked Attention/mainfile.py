# -*- coding: utf-8 -*-
"""
Created on Tue May 08 19:06:33 2018

author: Akshita Gupta
"""
from get_data import get_metadata, prepare_embeddings, get_train_data, get_test_data
from my_models import san_atten
import numpy as np

root_path                 = 'E:/akshita_workspace/cc/Audio-Vision/VQA/Stacked Attention/data/'
val_file                  = root_path + 'mscoco_val2014_annotations.json'
input_json                = root_path + 'vqa_data_prepro.json'   # Vocab and answers 
input_img_train_h5        = root_path + 'img_train_2.h5'   #Images train features
input_img_test_h5         = root_path + 'img_test_2.h5'   #Images test features
input_ques_h5             = root_path + 'vqa_data_prepro.h5' #Question features


common_word_emb_dim       = 512
word_emb_dim              = 300
embedding_matrix_filename = root_path + 'embeddings_%s.h5'%word_emb_dim
glove_path                = root_path + 'glove.6B.300d.txt'
img_norm                  = 1
nb_classes                = 1000
optimizer                 = 'sgd'
batch_size                = 20
epochs                    = 4
activation_1              = 'tanh'
activation_2              = 'tanh'
dropout                   = 0.5
vocabulary_size           = 15107
num_hidden_units_lstm     = 512
max_ques_length           = 26
num_hidden_layers_mlp     = 3
num_hidden_units_mlp      = 512
class_activation          = 'sigmoid'
img_vec_dim               = 4096
filter_sizes              = [128,256,256]
num_attention_layers      = 1


metadata=get_metadata(input_json)

#embedding_matrix = prepare_embeddings(vocabulary_size, word_emb_dim, metadata, embedding_matrix_filename, glove_path)
#print(embedding_matrix.shape)
model= san_atten(common_word_emb_dim,img_vec_dim, activation_1,activation_2, dropout, vocabulary_size,
                num_hidden_units_lstm, max_ques_length,
                word_emb_dim, num_hidden_layers_mlp,
                num_hidden_units_mlp, nb_classes, class_activation,filter_sizes,num_attention_layers)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary() # prints model layers with weights

train_X, train_Y= get_train_data(input_img_train_h5, input_ques_h5)

test_X,test_Y, multi_val_y=get_test_data(input_img_test_h5, input_ques_h5,metadata,val_file)

model.fit(train_X, train_Y, batch_size = batch_size, epochs=epochs, validation_data=(test_X, test_Y),verbose=1)

print ("Evaluating Accuracy on validation set:")
metric_vals = model.evaluate(test_X, test_Y)
print ("")
for metric_name, metric_val in zip(model.metrics_names, metric_vals):
    print (metric_name, " is ", metric_val)

# Comparing prediction against multiple choice answers
true_positive = 0
preds = model.predict(test_X)
pred_classes = [np.argmax(_) for _ in preds]
for i, _ in enumerate(pred_classes):
    if _ in multi_val_y[i]:
        true_positive += 1
print ("true positive rate: ", np.float(true_positive)/len(pred_classes))
