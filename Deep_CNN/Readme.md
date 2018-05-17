# A convolutional neural network approach for acoustic scene classification

*- Michele Valenti, Stefano Squartini, Aleksandr Diment, Giambattista Parascandolo and Tuomas Virtanen, IJCNN, 2017*[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7966035)][[Dataset](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging)]

## Train your own network

<div align=center>
	<img src="./cnn.PNG" width="800" >
</div>

### Make imports

Clone [Keras_aud](https://github.com/channelcs/keras_aud) and pass the cloned path to `ka_path`.

```
import sys
ka_path="path/to/keras_aud"
sys.path.insert(0, ka_path)
from keras_aud import aud_audio, aud_model, aud_utils
```

### Give paths for audio, features, csvs

We now give paths where
1. Audio is saved
2. Features need to be extracted
3. CSVs reside(present in keras_aud)

```
wav_dev_fd   = 'audio/dev'
wav_eva_fd   = 'audio/eva'
dev_fd       = 'features/dev'
eva_fd       = 'features/eva'
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

**LOGARITHM - MEL FILTER BANK** 

Pass `extract = True` to unpack the dataset into folders.

```python
aud_audio.extract('logmel', wav_dev_fd, dev_fd+'/logmel','yaml_file',dataset='chime_2016')
```

### Load Data

We now load the data and check their shape.

```python
tr_X, tr_y = GetAllData( dev_fd+'/logmel', meta_train_csv)
print(tr_X.shape)
print(tr_y.shape)    
```
*Output:*
```python
(11676L, 10L, 60L)
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
(11676L, 1L, 10L, 60L)
```

### Model

The Model here uses Deep-CNN with multiple layers to obtain feature maps which are followed by pooling and batch normalization.

```python
miz=aud_model.Functional_Model(model='CNN',dimx=dimx,dimy=dimy,num_classes=15,input_neurons=400,dropout=0.1,nb_filter=100,filter_length=3)
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
truth,pred=test(lrmodel,txt_eva_path)
acc=aud_utils.calculate_accuracy(truth,pred)
print "Accuracy %.2f prcnt"%acc
```
*Output:*
```
Accuracy:
```

### References
