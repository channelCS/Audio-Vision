<h2> A convolutional neural network approach for acoustic scene classification </h2>

*- Michele Valenti, Stefano Squartini, Aleksandr Diment, Giambattista Parascandolo and Tuomas Virtanen, IJCNN, 2017*
## Model
## Dependenices
This implementation uses Python 2.7, Keras 2.1 and Scikit Learn. The code works on Theano backend.
```
$ pip install requirements.txt
```
## Feature Extraction
- Methods Used
- Reason

## Training
- Dataset
    - All files are available to download from [here](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification). Extract the contents 
- Development Mode
- Evaluation Mode

## Results
- Dev :                                                         Eva: 

##References

##License
MIT





- Features
    - Log-mel features were chosen as a convenient image liked input for CNN. 
    - we calculate logmel features by measuring short-time Fourier transform (STFT) of raw audio with 40 ms window size and 50% overlap. Then, square the absolute value and combine them with 40 mel frequencies resulting in 40 mel energies. Finally, Logarithmic of the mel energies is calculated. The final result is further normalized by subtracting its mean and dividing it by its standard deviation.
    - For the proposed model, the author opted for further splitting the spectrograms into shorter, non-overlapping spectrograms which were called sequenced hereafter.

- Architecture proposed
    - The author proposed two-layer Deep-CNN with batch normalization resulting in the increase of model complexity and speed for adapting its own layer weights.
    - The structure of the model includes:
        - first convolution layer with 128 relu kernels with 5x5 filter size followed with polling layer.
        - second condition layer with 256 relu kernels followed with pooling layer.
        - Averaging all the prediction scores with argmax()
  
- Technique used
    - Deep-CNN with log-mel features for audio scene classification as log-mel showed notable audio spectra for a model to train.  





