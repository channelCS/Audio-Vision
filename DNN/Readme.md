# Deep Neural Network Baseline For Dcase Challenge 2016

*- Qiuqiang Kong, Iwnoa Sobieraj, Wenwu Wang, Mark Plumbley, Detection and Classification of Acoustic Scenes and Events, 2016* [[Paper](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/challenge_technical_reports/DCASE2016_Kong_3008.pdf)][[Dataset](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)]

## Train your own network

<div align=center>
	<img src="./DNN.png" width="500">
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
label_csv    = ka_path+'/keras_aud/utils/dcase16_task1/dev/meta.txt'
txt_eva_path = ka_path+'/keras_aud/utils/dcase16_task1/eva/test.txt'
new_p        = ka_path+'/keras_aud/utils/dcase16_task1/eva/evaluate.txt'
```

### Feature Extraction

**Logarithmic Mel Filter Bank** We have used this feature due to **GIVE REASON**

Pass `extract = True` to unpack the dataset into folders.

```python
aud_audio.extract('logmel', wav_dev_fd, dev_fd+'/logmel','yaml_file',dataset='dcase_2016')
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
miz=aud_model.Functional_Model(model='DNN',dimx=dimx,dimy=dimy,num_classes=15,act1='relu',act2='sigmoid',act3='relu',act4='softmax',input_neurons=500,dropout=0.1)
```

### Training

Pass `prep = 'dev'` to train on train and evaluate on val, and `prep = 'eval'` train on train+val and evaluate on test.

```python
lrmodel=miz.prepare_model()
lrmodel.fit(train_x,train_y,batch_size=100,epochs=10,verbose=1)    
```

### Results

We calculate `Accuracy` as the **FILL SOMETHING**.
 
```python
truth,pred=test(lrmodel,txt_eva_path)
acc=aud_utils.calculate_accuracy(truth,pred)
print "Accuracy %.2f prcnt"%acc
```
*Output:*
```
Accuracy 86.15%
```


## References

1. https://github.com/qiuqiangkong/DCASE2016_Task1


