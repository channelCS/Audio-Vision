# -*- coding: utf-8 -*-
"""
Created on Tue May 08 17:13:52 2018
author: @adityac8 @akshitac8
"""

import warnings
warnings.simplefilter("ignore")

import sys
ka_path="../dependencies"
sys.path.insert(0, ka_path)
from keras_aud import aud_model, aud_utils

#import my_models
import os
############# MAKE IMPORTS   #####################
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.utils import to_categorical
from get_data import GetValues
from sklearn.model_selection import train_test_split

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from file_logger import FileLogger
from time import time
import config as cfg
np.random.seed(68)

class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss']),
                           str(logs['acc']),
                           str(logs['val_acc'])])
############# GET MODEL PARAMETERS #################
seth = GetValues()
model='DNN'
prep='dev'
folds=4
##MAKE YOUR CHANGES
#dropout=0.1
#act!=linear

file_logger = FileLogger('out_{}.tsv'.format(model), ['step', 'tr_loss', 'te_loss',
                                                               'tr_acc', 'te_acc'])


############# EXTRACT FEATURES #####################

dropout, act1, act2,act3, input_neurons, epochs, batchsize, num_classes, _, _,\
loss,nb_filter, filter_length,pool_size, optimizer=seth.get_parameters()


############# LOAD DATA ###########################
tr_X, tr_y = seth.get_train_data('logmel_trainx2.cpkl')
v_X,  v_y = seth.get_val_data('logmel_testx2.cpkl')

dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
tr_X=aud_utils.mat_3d_to_nd(model,tr_X)
v_X=aud_utils.mat_3d_to_nd(model,v_X)


print("Functional_model {}".format(model))
print("Activation 1 {} 2 {} 3 {}".format(act1,act2,act3))
print("Dropout {}".format(dropout))
print("Kernels {} Size {} Poolsize {}".format(nb_filter,filter_length,pool_size))
print("Loss {} Optimizer {}".format(loss,optimizer))


#train_x=np.array(tr_X)
#train_y=np.array(tr_y)
#test_x=np.array(v_X)
#test_y=np.array(v_y)
##train_x, test_x, train_y, test_y = train_test_split( tr_X, tr_y, test_size=0.2, random_state=42)
#print("Evaluation mode")
#lrmodel=miz.prepare_model()
#train_y=to_categorical(train_y)
#test_y=to_categorical(test_y)
##fit the model
##slrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
#fold_='saved_models_2'
#if not os.path.exists(fold_):
#    os.mkdir(fold_)
#filepath=fold_+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
##tensorboard = TensorBoard(log_dir="logs/final/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)
#metrics_history = MetricsHistory()
#lrmodel.fit(x=train_x,
#          y=train_y,
#          batch_size=batchsize,
#          epochs=epochs,
#          verbose=1,
#          shuffle=True,
#          validation_data=(test_x, test_y),
#          callbacks=[metrics_history, reduce_lr, checkpoint])
#
#lrmodel.save(fold_+'/last-model-{0:02d}.hdf5'.format(epochs))
"""
truth,pred=seth.get_test_predictions(lrmodel,'mel_testx.cpkl')

eer=aud_utils.calculate_eer(truth,pred)

p,r,f=aud_utils.prec_recall_fvalue(pred,truth,0.4,'macro')
print("EER %.2f"%eer)
print("Precision %.2f"%p)
print("Recall %.2f"%r)
print("F1 score %.2f"%f)
file_=open('run1.txt','a')
str1="act3={}, input_neurons={}, batchsize={}, loss={}, nb_filter={}, filter_length={}, optimizer={}".format(act3, input_neurons, batchsize, loss,nb_filter, filter_length, optimizer)
str2="EER={0:.2f} Precision={1:.2f} Recall={2:.2f} F1 score={3:.2f}".format(eer,p,r,f)
file_.write(str1+'\n'+str2+'\n')
file_.close()
"""
#file_logger.close()

miz=aud_model.Functional_Model(input_neurons=input_neurons,dropout=dropout,
    num_classes=num_classes,model=model,dimx=dimx,dimy=dimy,
    nb_filter=nb_filter,act1=act1,act2=act2,act3=act3,
    filter_length=filter_length,pool_size=pool_size,
    optimizer=optimizer,loss=loss)
#fit the model
#tr_y=to_categorical(tr_y,len(cfg.labels))
#v_y=to_categorical(v_y,len(cfg.labels))
#fold_='saved_models_3'
#if os.path.exists(fold_):
#    os.rmdir(fold_)
#os.mkdir(fold_)
#filepath=fold_+"/weights-improvement1-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
#metrics_history = MetricsHistory()
#lrmodel=miz.prepare_model()
#lrmodel.fit(x=[tr_X,tr_X],
#            y=tr_X,
#            batch_size=128,epochs=50,verbose=2,
##          validation_data=(v_X, v_y)
#            #,callbacks=[metrics_history, reduce_lr, checkpoint]
#            )

if prep=='dev':
    kf = KFold(len(tr_X),folds,shuffle=True,random_state=42)
    results=[]    
    for train_indices, test_indices in kf:
        train_x = [tr_X[ii] for ii in train_indices]
        train_y = [tr_y[ii] for ii in train_indices]
        test_x  = [v_X[ii] for ii in test_indices]
        test_y  = [v_y[ii] for ii in test_indices]
        
        train_x=np.array(tr_X)
        train_y=np.array(tr_y)
        test_x=np.array(v_X)
        test_y=np.array(tr_y)
        train_y = to_categorical(train_y,num_classes=len(cfg.labels))
        test_y = to_categorical(test_y,num_classes=len(cfg.labels)) 
        print ("Development Mode")
        
        #get compiled model
        lrmodel=miz.prepare_model()
        
        if lrmodel is None:
            print("If you have used Dynamic Model, make sure you pass correct parameters")
            raise SystemExit
        
        #        lrmodel.save(fold_+'/last-model-{0:02d}.hdf5'.format(epochs))
        lrmodel.fit(train_x, batch_size=128,epochs=100,verbose=2,validation_data=(test_x, None))


        #make prediction
        #xx=np.zeros(test_x.shape)
        #pred=lrmodel.predict([test_x,xx])
        pred=lrmodel.predict(test_x)
        
        #bre
        pred = [ii.argmax()for ii in pred]
        test_y = [ii.argmax()for ii in test_y]
        
        
        results.append(accuracy_score(pred,test_y))
        print accuracy_score(pred,test_y)
        print accuracy_score(pred,test_x)
        jj=str(set(list(test_y)))
        print "Unique in test_y",jj

    

print "Results: " + str( np.array(results).mean() )
#elif prep=='eval':
#    train_x=np.array(tr_X)
#    train_y=np.array(tr_y)
#    print "Evaluation mode"
#    lrmodel=miz.prepare_model()
#        
#    #fit the model
#    lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
#    
#    truth,pred=seth.get_test_predictions(lrmodel)
#
#    eer=aud_utils.calculate_eer(truth,pred)
#    
#    p,r,f=aud_utils.prec_recall_fvalue(pred,truth,0.4,'macro')
#    print "EER %.2f"%eer
#    print "Precision %.2f"%p
#    print "Recall %.2f"%r
#    print "F1 score %.2f"%f
#else:
#    raise Exception('Wrong prep',prep)
#
#import csv
#import cPickle
#import scipy
#
#csv_file='C:/users/akshita/downloads/kaggle/test.txt'
#with open( csv_file, 'rb') as f:
#    reader = csv.reader(f)
#    lis = list(reader)
#
#pred_lbs=[]
#x3 = cPickle.load(open('logmel_kaggle.cpkl', 'rb'))
#for li in lis:
#    na = li[0][6:-4]
#    #audio evaluation name
#    X0 = x3[na]
#    X0 = aud_utils.mat_2d_to_3d( X0, 10, 10)
#    
#    X0 = aud_utils.mat_3d_to_nd(model,X0)
#
#    # predict
#    p_y_preds = lrmodel.predict(X0)        # probability, size: (n_block,label)
#    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
#    b = scipy.stats.mode(preds)
#    pred = int( b[0] )
#    pred_lbs.append( cfg.id_to_lb[ pred ] )
#
#file_=open('kag.csv','w')
#file_.write('Id,Scene_label\n')
#for i in range(len(pred_lbs)):
#    file_.write(str(i)+','+pred_lbs[i]+'\n')
#file_.close()
