# Deep Neural Network Baseline For Dcase Challenge 2016

*- Qiuqiang Kong, Iwnoa Sobieraj, Wenwu Wang, Mark Plumbley, Detection and Classification of Acoustic Scenes and Events, 2016*
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

## References

## License
MIT





- Features
    -  Mfcc are widely used for speech recognition but it looses essential part of Information required for Classification.
    -  For, this task they have chosen Mel filter bank over other features as it showed prominent results over others
    -  They have computed mel-filterbanks on the basis of same height and area.
    -  Mel was able to detect classes with frequencies > 1KHz
- Architecture proposed
    - Deep Neural Network (DNN) with 400 input nodes (10 frames * 40 mel frequencies) and 500 hidden units per layer with 3 hidden layers.
  
- Technique used
    - The paper proposed same DNN architecture for the task of Audio classification, multi-label classification, and, event detection for different scenes.
    - Deep Neural Network, when ued with mel spectrogram, proves to be a novel idea for audio-processing because of its ability to learn, generalise and predict on the basis of the pattern.





