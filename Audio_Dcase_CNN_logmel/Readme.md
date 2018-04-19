# Deep Neural Network Baseline For Dcase Challenge 2016

*- Qiuqiang Kong, Iwnoa Sobieraj, Wenwu Wang, Mark Plumbley, Detection and Classification of Acoustic Scenes and Events, 2016*

## Summary

The paper introduces Deep Neural Network approach for Audio processing and acts as a baseline for all the task of Dcase challenge 2016 :

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

## Strengths
- The code provided by the author is very easily reproducible for further experimentations.
- Sets a standard for implementations of Deep models with mel spectrograms and other features such as CQT.

## Weaknesses / Notes
-  Paper showed lack of parameter tuninng for feature extraction as well as Deep models.
-  We can propose the use of recurrent models for the mentioned tasks.
- The paper feels a little half-baked in parts, and some ideas could've been presented more clearly.


This is an implementation of DEEP NEURAL NETWORK BASELINE FOR DCASE CHALLENGE 2016 described in the following paper : *Qiuqiang Kong, Iwona Sobieraj, Wenwu Wang, Mark D. Plumbley. [DEEP NEURAL NETWORK BASELINE](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/workshop/Kong-DCASE2016workshop.pdf)*. 

![Model architecture](https://github.com/akshitac8/Paper-Implementation/blob/master/Audio/Dcase16Task1_Kong/Dcase_kong.png)

## Code used is a modification from [Dcase](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/workshop/Kong-DCASE2016workshop.pdf)

## Dependencies
This implementation uses Python 2.7, Keras 2.1 and Scikit Learn. The code works on Theano backend. 

## Dataset 
Please download the dataset from [DCASE website](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification). Extract the contents and keep it in the folder as specified in **config** file.