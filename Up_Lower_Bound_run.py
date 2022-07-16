import keras
import os, sys
import pickle
import pandas as pd
from keras import backend as K
from keras import optimizers
import tensorflow as tf 

from Up_Lower_Bound import UpperLowerBound
from tensorflow.keras import callbacks

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

tf.random.set_seed(1)

class Run:

   No_PCGrad = UpperLowerBound()
   PCGrad = UpperLowerBound()

   epochs = 2000
   batch_size = 256

   def __init__(self):

      self.No_PCGrad_model = Run.No_PCGrad.model 
      self.PCGrad_model = Run.PCGrad.model

      self.result = []
      self.opt = optimizers.Adam()

   @classmethod
   def set_epochs(cls, epoches):
      cls.epochs = epoches

   @classmethod
   def set_batch_size(cls, batch_size):
      cls.batch_size = batch_size

   def run_no_pcgrad(self):
      
      self.No_PCGrad_model.init_arguments()

      early_stopping = callbacks.EarlyStopping(
         monitor = 'mpiw',
         min_delta=0.0000000001,  # an absolute change of less than min_delta, will count as no improvement.
         patience=500,  # Number of epochs with no improvement after which training will be stopped
         restore_best_weights=True
      )

      self.No_PCGrad_model.compile(optimizer=self.opt,
      loss = [self.No_PCGrad_model.selective_up, self.No_PCGrad_model.selective_low, self.No_PCGrad_model.up_penalty, self.No_PCGrad_model.low_penalty, self.No_PCGrad_model.coverage_penalty],
      metrics = [self.No_PCGrad_model.coverage, self.No_PCGrad_model.mpiw])

      history_no_pcgrad = self.No_PCGrad_model.fit(Run.No_PCGrad.X_train, Run.No_PCGrad.y_train, 
      validation_data = (Run.No_PCGrad.X_test, [Run.No_PCGrad.y_test[:,0], Run.No_PCGrad.y_test[:,1], Run.No_PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[early_stopping], 
      verbose=1)

      # Save the training history 
      with open('history_no_pcgrad_history.pkl', 'wb') as handle:
         pickle.dump(history_no_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

      self.plot_training('history_no_pcgrad_history.pkl')

      no_pcgrad_pred = self.No_PCGrad_model.predict(Run.No_PCGrad.X_test)

      df = pd.DataFrame(no_pcgrad_pred, columns = ['Lowerbound', 'Upbound', 'Coverage_rate'])
      df['y_true'] = Run.No_PCGrad.y_test[:,0]
      df['Width'] = (df['Upbound']-df['Lowerbound'])
      df['NMPIW'] = (df['Upbound']-df['Lowerbound'])/ Run.No_PCGrad.range
      #df['Flag']= np.where((df['Upbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)  

      self.result.append({'Coverage_rate':np.mean(df['Coverage_rate']), 'NMPIW':np.mean(df['NMPIW'])})
      
      df.to_csv('no_pcgrad_pred.csv')
   

   def run_pcgrad(self):
      self.PCGrad_model.init_arguments(method = 'PCGrad')

      self.PCGrad_model.compile(optimizer=self.opt,
      loss = [self.PCGrad_model.selective_up, self.PCGrad_model.selective_low, self.PCGrad_model.up_penalty, self.PCGrad_model.low_penalty, self.PCGrad_model.coverage_penalty],
      metrics = [self.PCGrad_model.coverage, self.PCGrad_model.mpiw])

      history_pcgrad = self.PCGrad_model.fit(Run.PCGrad.X_train, Run.PCGrad.y_train, 
      validation_data = (Run.PCGrad.X_test, [Run.PCGrad.y_test[:,0], Run.PCGrad.y_test[:,1], Run.PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[early_stopping], 
      verbose=1)

      with open('history_pcgrad_history.pkl', 'wb') as handle:
         pickle.dump(history_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)     
      #model.save_weights("checkpoints/{}".format(self.filename))

      self.plot_training('history_pcgrad_history.pkl')
      pcgrad_pred = self.PCGrad_model.predict(Run.PCGrad.X_test)

      df = pd.DataFrame(pcgrad_pred, columns = ['Lowerbound', 'Upbound', 'Coverage_rate'])
      df['y_true'] = Run.PCGrad.y_test[:,0]
      df['Width'] = (df['Upbound']-df['Lowerbound'])
      df['NMPIW'] = (df['Upbound']-df['Lowerbound'])/ Run.No_PCGrad.range
      #df['Flag']= np.where((df['Upbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0) 

      self.result.append({'Coverage_rate':np.mean(df['Coverage_rate']), 'NMPIW':np.mean(df['NMPIW'])})

      #Save predicted values 
      df.to_csv('pcgrad_pred.csv')

   def print_comparison(self):
      res = pd.DataFrame(self.result, index = ['No_pcgrad', 'With_pcgrad'])
      print(res)

   def plot_training(self, filename):
      dict_data = pd.read_pickle(filename)  
      df = pd.DataFrame(dict_data)
      title = '-'.join(filename.split('_')[:-1])
      fig = plt.figure(figsize=(10,6))
      sns.set_style("ticks")
      plt.title(title)
      plt.xlabel("Epochs")
      sns.lineplot(data=df[ ['coverage', 'mpiw', 'val_coverage', 'val_mpiw','val_loss']])
      plt.savefig(f'{title}.png', dpi = 600)
      plt.clf()
      # plt.show()


if __name__ == "__main__":
   obj = Run()

   obj.run_no_pcgrad()
   obj.run_pcgrad()
   obj.print_comparison()

   #Run.epochs = 2500
   # obj.run_no_pcgrad()
   # obj.run_pcgrad()
   # obj.print_comparison()











