import numpy as np 
import pandas as pd  
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from tensorflow.keras import layers
from UpperLower_Control import UpperLower_Control 

class UpperLowerBound:

	filepath= 'dataset/'
	
	def __init__(self, dataset_name='Concrete_Data.xls'):

		self.dataset_name = dataset_name 
		self._load_data()
		self.model = self.build_model()

	# Custom a training model 
	def build_model(self):

		inputs = Input(shape=self.X_train.shape[1:])
		curr = layers.Dense(64, activation='relu', kernel_initializer='normal')(inputs) 
		curr = layers.Dense(32, activation='relu', kernel_initializer='normal')(curr)

		# Lower bound  (head)
		low_bound = layers.Dense(8)(curr) 
		low_bound = layers.Dense(1, name = 'upper_bound')(low_bound) 

		# Upper bound  (head)
		up_bound = layers.Dense(8)(curr)    
		up_bound = layers.Dense(1, name = "lower_bound")(up_bound)

		# Selective (head)
		selective = layers.Dense(16, activation='relu', kernel_initializer='normal')(curr)
		selective = Dense(1, activation='sigmoid')(selective)

		combined_outputs = Concatenate(axis=1, name="combined_output") ([low_bound, up_bound, selective])

		return UpperLower_Control(inputs, combined_outputs)

	# y is the target value 
	def _load_data(self, y = 'Concrete compressive strength(MPa, megapascals) '):

		file_path = os.path.join(UpperLowerBound.filepath, self.dataset_name)

		if file_path.split('.')[-1] == 'xls' or file_path.split('.')[-1] == 'xlsx' :
			df_data = pd.read_excel(file_path)
		else:
		 	df_data = pd.read_csv(file_path)

		X = df_data.drop(y, axis = 1)
		y = df_data[y].values.reshape(-1,1)

		# Scale data to [0,1]
		X, y = self.scaled_data(X, y)

        # Split data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		y_train = y_train.astype('float32')
		y_test = y_test.astype('float32')

		# Change y based on the model 
		self.y_train = np.repeat(y_train, [2], axis = 1)
		self.y_test = np.repeat(y_test, [2], axis = 1)
		self.y_train = np.hstack((self.y_train, np.zeros((self.y_train.shape[0], 1), dtype=self.y_train.dtype)))
		self.y_test = np.hstack((self.y_test, np.zeros((self.y_test.shape[0], 1), dtype=self.y_test.dtype)))

		self.range = max(y) - min(y)

		self.X_train, self.X_test = X_train,  X_test

		# Scale data to [0,1]
	def scaled_data(self, X, y):
		X = MinMaxScaler().fit_transform(X)
		y = MinMaxScaler().fit_transform(y)
		return X, y 

	def predict(self, x=None, batch_size=256):
		if x is None:
			x = self.X_test
		return self.model.predict(x, batch_size)






	





















