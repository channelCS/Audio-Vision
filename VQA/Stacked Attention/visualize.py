# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 03:19:10 2018

@author: akshita
"""
import sys
import warnings
import os, argparse

warnings.simplefilter("ignore")

import spacy
import cv2, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_image_model():
    ''' Takes the CNN weights file, and returns the VGG model update 
    with the weights. Requires the file VGG.py inside models/CNN '''
#    from models.CNN.VGG import VGG_16
#    image_model = VGG_16(CNN_weights_file_name)
    
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)  
    
    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
#    image_features = np.zeros((1, 4096))
    image_features = np.zeros((1,4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im=cv2.imread(image_file_name)
    if im is None:
        raise Exception("Incorrect path")
#    cv2.imshow('Image',im)
#    cv2.waitKey(0)

    im = cv2.resize(im, (224,224)).astype(np.float32)
    
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]
    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 
    x = preprocess_input(im)

    image_features[0,:] = get_image_model().predict(x)[0]
    return image_features

def get_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
#    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

#word_embeddings = spacy.load('en', vectors='en_glove_cc_☺300_1m_vectors')

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''
    # thanks the keras function for loading a model from JSON, this becomes
    # very easy to understand and work. Alternative would be to load model
    # from binary like cPickle but then model would be obfuscated to users
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

#from keras.utils.visualize_util import plot
VQA_model_file_name      = 'data/vqa_aditya_model.json'
VQA_weights_file_name   = 'data/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'data/labelencoder_trainval.pkl'
#CNN_weights_file_name   = 'vgg19_weights.h5
model_vqa = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
#plot(model_vqa, to_file='model_vqa.png')

def plot(image_file_name):
    img=mpimg.imread(image_file_name)
    imgplot = plt.imshow(img)
    plt.show()

def predict_answer(question):
    image_features = get_image_features(image_file_name)
    question_features = get_question_features(question)
    
    y_output = model_vqa.predict([question_features, image_features])

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of training and thus would 
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(np.argsort(y_output)[0,-5:]):
        print (str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))
       
image_file_name = 'data/coco/COCO_test2015_000000000554.jpg'
plot(image_file_name)
questions = ["Is there coffee?","how many?","what gender?"]
for question in questions:
    print(question)
    predict_answer(question)
    