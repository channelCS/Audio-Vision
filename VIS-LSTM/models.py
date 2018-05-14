import numpy as np
import embedding
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Merge, Reshape, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten

def vis_lstm():
	embedding_matrix = embedding.load()
	inpx0=Input(shape=(embedding_matrix.shape[0],embedding_matrix.shape[1]))
	x0=Embedding(weights = [embedding_matrix],trainable = False)(inpx0)
	###Make embedding_model
	embedding_model = Model([inpx0],x0)
	
	inpx1=Input(shape=(4096,))
	x1=Dense(embedding_matrix.shape[1],activation='linear')(inpx1)
	x1=Reshape((1,embedding_matrix.shape[1]))(x1)
	###Make image_model
	image_model = Model([inpx1],x1)

	inpx2=Merge([image_model,embedding_model],mode = 'concat',concat_axis = 1))
	x2=LSTM(1001)(inpx2)
	x2=Dropout(0.5)(x2)
	x2=Dense(1001,activation='softmax')(x2)
	###Make main_model
	main_model = Model([inpx2],x2)
	
	return main_model

def vis_lstm_2():
	embedding_matrix = embedding.load()
	inpx0=Input(shape=(embedding_matrix.shape[0],embedding_matrix.shape[1]))
	x0=Embedding(weights = [embedding_matrix],trainable = False)(inpx0)
	###Make embedding_model
	embedding_model = Model([inpx0],x0)

	inpx1=Input(shape=(4096,))
	x1=Dense(embedding_matrix.shape[1],activation='linear')(inpx1)
	x1=Reshape((1,embedding_matrix.shape[1]))(x1)
	###Make image_model_1
	image_model_1 = Model([inpx1],x1)

	inpx2=Input(shape=(4096,))
	x2=Dense(embedding_matrix.shape[1],activation='linear')(inpx2)
	x2=Reshape((1,embedding_matrix.shape[1]))(x2)
	###Make image_model_2
	image_model_1 = Model([inpx2],x2)

	inpx3=Merge([image_model_1,embedding_model,image_model_2],mode = 'concat',concat_axis = 1))
	x3=LSTM(1001)(inpx3)
	x3=Dropout(0.5)(x3)
	x3=Dense(1001,activation='softmax')(x3)
	###Make main_model
	main_model = Model([inpx3],x3)

	
	return main_model

def VGG_16(weights_path=None):
	inpx0=Input(shape=(3,224,224))
	x=ZeroPadding2D((1,1))(inpx0)
	x=Conv2D(64, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(64, 3, 3, activation='relu')(x)
	x=MaxPooling2D((2,2), strides =(2,2))(x)

	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(128, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(128, 3, 3, activation='relu')(x)
	x=MaxPooling2D((2,2), strides =(2,2))(x)

	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(256, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(256, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(256, 3, 3, activation='relu')(x)
	x=MaxPooling2D((2,2), strides =(2,2))(x)

	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=MaxPooling2D((2,2), strides =(2,2))(x)

	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=ZeroPadding2D((1,1))(x)
	x=Conv2D(512, 3, 3, activation='relu')(x)
	x=MaxPooling2D((2,2), strides =(2,2))(x)

	x=Flatten()(x)
	x=Dense(4096, activation='relu')(x)
	x=Dropout(0.5)(x)
	x=Dense(4096, activation='relu')(x)
	x=Dropout(0.5)(x)
	x=Dense(1000, activation='softmax')(x)


	#Make model
	model = Model([inpx0],x)

	if weights_path:
		model.load_weights(weights_path)
	
	return model
