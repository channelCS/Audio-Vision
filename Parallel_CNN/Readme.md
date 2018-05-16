# Cqt-based Convolutional Neural Networks for Audio Scene Classification and Domestic Audio Tagging

*- Thomas Lidy, Alexander Schindler, Detection and Classification of Acoustic Scenes and Events, 2016*[[Paper](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/challenge_technical_reports/DCASE2016_Lidy_4007.pdf)][[Dataset1](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)]
[[Dataset2](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging)]

## Model
The Model here uses parrallel CNN architecture to capture relevant feature maps for both time and frequency domain.

## Train your own network

<div align=center>
	<img src="./cqt_cnn.PNG" width="738">
</div>

### Make imports

Clone [Keras_aud](https://github.com/channelcs/keras_aud) and pass the cloned path to `ka_path`.

```
import sys
ka_path="path/to/keras_aud"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_model, aud_utils
```

### Give paths for audio, and features

We now give paths where
1. Audio is saved
2. Features need to be extracted

```
wav_dev_fd   = 'audio/dev'
wav_eva_fd   = 'audio/eva'
dev_fd       = 'features/dev'
eva_fd       = 'features/eva'
```

## Acoustic Scene Classification

### Feature Extraction

**Constant Q Transform** We have used this feature due to **GIVE REASON**

Pass `extract = True` to unpack the dataset into folders.

```python
aud_audio.extract('cqt', wav_dev_fd, dev_fd+'/cqt','yaml_file',dataset='dcase_2016')
```

### Load Data

We now load the data and check their shape.

```python
tr_X, tr_y = GetAllData( dev_fd+'/cqt', meta_train_csv)
print(tr_X.shape)
print(tr_y.shape)    
```
*Output:*
```python
(11676L, 10L, 40L)
(11676L, 8L)
```
We take the last two dimensions which act as the `Input` shape for our `DNN` model.
```python
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
```

We need to pass a 2D array to our CNN model. We reshape our model using
```python
tr_X=aud_utils.mat_3d_to_nd('DNN',tr_X)
print(tr_X.shape)
```
*Output:*
```python
(11676L, 400L)
```

## Model
The model here uses mel bank features with Deep Neural Network.

```python
miz=aud_model.Functional_Model(model='CNN',dimx=dimx,dimy=dimy,num_classes=15,act1='relu',act2='relu',act3='softmax',input_neurons=500,dropout=0.25,nb_filter=100,filter_length=3)
```

### Training

Pass `prep = 'dev'` to train on train and evaluate on val, and `prep = 'eval'` train on train+val and evaluate on test.

```python
lrmodel=miz.prepare_model()
lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)    
```

### Results

We calculate `Accuracy` as the **FILL SOMETHING**.
 
```python
truth,pred=test(lrmodel,txt_eva_path)
acc=aud_utils.calculate_accuracy(truth,pred)
print "Accuracy %.2f prcnt"%acc
```

## Audio Tagging

### Give paths for csvs

We now give paths where CSVs reside(present in keras_aud)

```
meta_train_csv  = ka_path+'/keras_aud/utils/dcase16_task4/meta_csvs/development_chunks_refined.csv'
meta_test_csv   = ka_path+'/keras_aud/utils/dcase16_task4/meta_csvs/evaluation_chunks_refined.csv'
label_csv       = ka_path+'/keras_aud/utils/dcase16_task4/label_csvs'
```

### Preprocess CHiME dataset

Pass `unpack = True` to unpack the dataset into folders.

```python
aud_utils.unpack_chime_2k16('path/to/chime_home',wav_dev_fd,wav_eva_fd,meta_train_csv,meta_test_csv,label_csv)
```

### Feature Extraction

**Constant Q Transform** We have used this feature due to **GIVE REASON**

Pass `extract = True` to unpack the dataset into folders.

```python
aud_audio.extract('cqt', wav_dev_fd, dev_fd+'/cqt','yaml_file',dataset='chime_2016')
```

### Load Data

We now load the data and check their shape.

```python
tr_X, tr_y = GetAllData( dev_fd+'/cqt', meta_train_csv)
print(tr_X.shape)
print(tr_y.shape)    
```
*Output:*
```python
(11676L, 10L, 40L)
(11676L, 8L)
```
We take the last two dimensions which act as the `Input` shape for our `CNN` model.
```python
dimx=tr_X.shape[-2]
dimy=tr_X.shape[-1]
```

We need to pass a 4D array to our CNN model. We reshape our model using
```python
tr_X=aud_utils.mat_3d_to_nd('CRNN',tr_X)
print(tr_X.shape)
```
*Output:*
```python
(11676L, 1L, 10L, 40L)
```

### Model

The model here extracts feature maps using convolution layers followed by pooling and stacked recurrent layers(lstm/GRU). We create a class instance and save it in `miz`. We use `softmax` as the last activation because **GIVE REASON**

```python
miz=aud_model.Functional_Model(model='TCNN',dimx=dimx,dimy=dimy,num_classes=15,act1='relu',act2='relu',act3='sigmoid',input_neurons=500,dropout=0.1,nb_filter=100,filter_length=3)
```

### Training

Pass `prep = 'dev'` to train on train and evaluate on val, and `prep = 'eval'` train on train+val and evaluate on test.

```python
lrmodel=miz.prepare_model()
lrmodel.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)    
```

### Results

We calculate `Equal Error Rate`, `Precision`,`Recall` and `F1-Score`. We use a `threshold = 0.4` and `macro` to get mean values.
 
```python
truth,pred=test(lrmodel,meta_test_csv,model)
eer=aud_utils.calculate_eer(truth,pred)
p,r,f=aud_utils.prec_recall_fvalue(pred,truth,0.4,'macro')
print "EER %.2f"%eer
print "Precision %.2f"%p
print "Recall %.2f"%r
print "F1 score %.2f"%f
```
*Output:*
```
Mainfile1.py
Accuracy: 84.5%

Mainfile2.py
EER : 0.15

```


### References
