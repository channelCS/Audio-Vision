"""
Created on Sat Apr 08 11:48:18 2018
@author: Akshita Gupta
Email - akshitadvlp@gmail.com

Updated on 15/04/18
@author: Aditya Arora
Email - adityadvlp@gmail.com
"""

import model_description as M
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import time
import csv
import cPickle
import scipy

np.random.seed(1234)

import numpy as np
import sys
import scipy

from keras import backend as K
K.set_image_dim_ordering('th')
   
class Functional_Model:
    """
    Class for functional model.
    
    Supported Models
    ----------
    DNN      : Deep Neural Network

    CNN      : Convolution Neural Network
    
    RNN      : Recurrent Neural Network
    
    CRNN     : Convolution Recurrent Neural Network
    
    CBRNN    : Bi-Directional Convolution Recurrent Neural Network
    
    MultiCNN : Multi Feature Convolution Neural Network
    
    ACRNN    : Attention Based Convolution Recurrent Neural Network
    
    TCNN     : Transpose Convolution Neural Network


    
    Parameters
    ----------
    model : str
        Name of Model
    dimx : int
        Second Last Dimension
    dimy : int
        Last Dimension
    num_classes : int
        Number of Classes
        
    Returns
    -------
    Functional Model
    """
    def __init__(self,model,dimx,dimy,num_classes,**kwargs):
        if model is None:
            raise ValueError("No model passed")
        self.model=model
        self.dimx = dimx
        self.dimy = dimy
        self.num_classes=num_classes
        self.kwargs=kwargs

    def prepare_model(self):
        """
        This function
        """
        if self.model=='DNN':
            lrmodel=M.dnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='CNN':
            lrmodel=M.cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='RNN':
            lrmodel=M.rnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs = self.kwargs)
        elif self.model=='CRNN':
            lrmodel=M.cnn_rnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='FCRNN':
            lrmodel=M.feature_cnn_rnn(num_classes = self.num_classes, dimx = self.dimx,dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='CBRNN':
            lrmodel=M.cbrnn(num_classes = self.num_classes, dimx = self.dimx,dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='ParallelCNN':
            lrmodel=M.parallel_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='MultiCNN':
            lrmodel=M.multi_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='ACRNN':
            lrmodel=M.ACRNN(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='TCNN':
            lrmodel=M.transpose_cnn(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='MultiACRNN':
            lrmodel=M.multi_ACRNN(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        elif self.model=='seq2seq':
            lrmodel=M.seq2seq(num_classes = self.num_classes, dimx = self.dimx, dimy = self.dimy, kwargs=self.kwargs)
        
        else:
            raise ValueError("Could not find model {}".format(self.model))
        return lrmodel
         
class Static_Model:
    def __init__(self,input_neurons,cross_validation,
        dimx,dimy,num_classes,
        nb_filter,filter_length,
        model):
        if model is None:
            raise ValueError("No model passed")
        self.cross_validation=cross_validation
        self.input_neurons=input_neurons
        self.model=model
        self.nb_filter = nb_filter
        self.filter_length =filter_length
        self.dimx = dimx
        self.dimy = dimy
        self.num_classes=num_classes

    def prepare_model(self):
        if self.model=='CHOU':
            lrmodel=M.conv_deconv_chou(dimx=self.dimx,dimy=self.dimy,nb_filter=self.nb_filter,num_classes=self.num_classes)
        else:
            raise ValueError("Could not find model {}".format(self.model))
        return lrmodel
                
class Dynamic_Model:
    def __init__(self,model,num_classes,dimx,dimy,acts,**kwargs):
        if model is None:
            raise ValueError("No model passed")
        self.model=model
        self.num_classes=num_classes
        self.dimx = dimx
        self.dimy = dimy
        self.acts = acts
        self.kwargs=kwargs
    def prepare_model(self):
        try:
            if self.model=='DNN':
                lrmodel=M.dnn_dynamic(num_classes = self.num_classes,
                                      input_dim   = self.dimx*self.dimy,
                                      acts        = self.acts,
                                      kwargs      = self.kwargs)
            elif self.model=='CNN':
                lrmodel=M.cnn_dynamic(num_classes = self.num_classes,
                                      dimx        = self.dimx,
                                      dimy        = self.dimy,
                                      acts        = self.acts,
                                      kwargs      = self.kwargs)
            elif self.model=='CBRNN':
                lrmodel=M.cbrnn_dynamic(num_classes = self.num_classes,
                                        dimx        = self.dimx,
                                        dimy        = self.dimy,
                                        acts        = self.acts,
                                        kwargs      = self.kwargs)
            else:
                raise ValueError("Could not find model {}".format(self.model))
            return lrmodel
        except Exception as e:
            print(e)
          
