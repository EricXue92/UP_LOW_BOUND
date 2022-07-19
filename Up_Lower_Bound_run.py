import os
import sys
import csv 

import pickle
import gc 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
 

from Up_Lower_Bound import UpperLowerBound
from tensorflow.keras import callbacks


# tf.random.set_seed(2)

class Run:

   epochs = 5
   batch_size = 256

   #Stop training when a monitored metric has stopped improving
   early_stopping = callbacks.EarlyStopping(
      monitor = 'coverage_width_rate',
      min_delta=0.0001,  # an absolute change of less than min_delta, will count as no improvement.
      patience=30,      # 0.1* epoches Number of epochs with no improvement after which training will be stopped
      restore_best_weights=True
   )

   def __init__(self, No_PCGrad, PCGrad):

      self.No_PCGrad = No_PCGrad
      self.PCGrad = PCGrad

      self.No_PCGrad_model = No_PCGrad.model 
      self.PCGrad_model = PCGrad.model

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

      self.No_PCGrad_model.compile(optimizer=self.opt,
      loss = [self.No_PCGrad_model.selective_up, self.No_PCGrad_model.selective_low, self.No_PCGrad_model.up_penalty, self.No_PCGrad_model.low_penalty, self.No_PCGrad_model.coverage_penalty],
      metrics = [self.No_PCGrad_model.coverage, self.No_PCGrad_model.mpiw, self.No_PCGrad_model.coverage_width_rate])

      history_no_pcgrad = self.No_PCGrad_model.fit(self.No_PCGrad.X_train, self.No_PCGrad.y_train, 
      validation_data = (self.No_PCGrad.X_test, [self.No_PCGrad.y_test[:,0], self.No_PCGrad.y_test[:,1], self.No_PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[Run.early_stopping], 
      verbose=0)

      # Save the training history 
      # with open(f'{self.No_PCGrad.dataset_name}_no_pcgrad_history.pkl', 'wb+') as handle:
      #    pickle.dump(history_no_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

      # self.plot_training(f'{self.No_PCGrad.dataset_name}_no_pcgrad_history.pkl')

      no_pcgrad_pred = self.No_PCGrad_model.predict(self.No_PCGrad.X_test)

      df = pd.DataFrame(no_pcgrad_pred, columns = ['Lowerbound', 'Upbound', 'Coverage_rate'])

      df['y_true'] = self.No_PCGrad.y_test[:,0]
      df['Width'] = (df['Upbound']-df['Lowerbound'])
      df['NMPIW'] = (df['Upbound']-df['Lowerbound'])/ self.No_PCGrad.range
      df['Coverage_rate/Width'] = df['Coverage_rate']/df['Width']

      df = df[['Lowerbound','y_true','Upbound','Coverage_rate','Width','NMPIW','Coverage_rate/Width']]
      #df['Flag']= np.where((df['Upbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)  

      # Save all predicted value
      # with open(f'{self.No_PCGrad.dataset_name}_no_pcgrad_pred.csv', 'a') as f:
      #    df.to_csv(f, header = True )

      self.result.append({'Coverage_rate':np.mean(df['Coverage_rate']), 'NMPIW':np.mean(df['NMPIW']),'Coverage_rate/Width':np.mean(df['Coverage_rate/Width'])})

   def run_pcgrad(self):
      self.PCGrad_model.init_arguments(method = 'PCGrad')

      self.PCGrad_model.compile(optimizer=self.opt,
      loss = [self.PCGrad_model.selective_up, self.PCGrad_model.selective_low, self.PCGrad_model.up_penalty, self.PCGrad_model.low_penalty, self.PCGrad_model.coverage_penalty],
      metrics = [self.PCGrad_model.coverage, self.PCGrad_model.mpiw, self.PCGrad_model.coverage_width_rate])

      history_pcgrad = self.PCGrad_model.fit(self.PCGrad.X_train, self.PCGrad.y_train, 
      validation_data = (self.PCGrad.X_test, [self.PCGrad.y_test[:,0], self.PCGrad.y_test[:,1], self.PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[Run.early_stopping], 
      verbose=0)

      # Save the training history 
      # with open(f'{self.PCGrad.dataset_name}_pcgrad_history.pkl', 'wb+') as handle:
      #    pickle.dump(history_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)     
      #model.save_weights("checkpoints/{}".format(self.filename))

      #self.plot_training(f'{self.PCGrad.dataset_name}_pcgrad_history.pkl')

      pcgrad_pred = self.PCGrad_model.predict(self.PCGrad.X_test)

      df = pd.DataFrame(pcgrad_pred, columns = ['Lowerbound', 'Upbound', 'Coverage_rate'])

      df['y_true'] = self.PCGrad.y_test[:,0]
      df['Width'] = (df['Upbound']-df['Lowerbound'])
      df['NMPIW'] = (df['Upbound']-df['Lowerbound'])/ self.No_PCGrad.range
      df['Coverage_rate/Width'] = df['Coverage_rate']/df['Width']

      df = df[['Lowerbound','y_true','Upbound','Coverage_rate','Width','NMPIW','Coverage_rate/Width']]
      #df['Flag']= np.where((df['Upbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0) 

      # Save all predicted value 
      # with open(f'{self.PCGrad.dataset_name}_pcgrad_pred.csv', 'a') as f:
      #    df.to_csv(f, header=True)


      self.result.append({'Coverage_rate':np.mean(df['Coverage_rate']), 'NMPIW':np.mean(df['NMPIW']),'Coverage_rate/Width':np.mean(df['Coverage_rate/Width']) })

   def print_comparison(self):
      res = pd.DataFrame(self.result, index = ['No_PCGrad', 'PCGrad'])
      print(res)
      return res

   # def plot_training(self, filename):
   #    dict_data = pd.read_pickle(filename)  
   #    df = pd.DataFrame(dict_data)
   #    title = f'{filename}'
   #    fig = plt.figure(figsize=(10,6))
   #    sns.set_style("ticks")
   #    plt.title(title)
   #    plt.xlabel("Epochs")
   #    sns.lineplot(data=df[ ['coverage', 'mpiw','val_coverage', 'val_mpiw','val_loss']])
   #    plt.savefig(f'{title}.png', dpi = 600)
   #    plt.clf()
   #    plt.show()
     

def main():

   # UpperLowerBound.dataset_name, UpperLowerBound.target  = '2_nonconstant_noise.csv', 'y'
   # obj.run_no_pcgrad()
   # obj.run_pcgrad()
   # obj.print_comparison()

   dataset_names = ['1_constant_noise.csv','2_nonconstant_noise.csv','4_Concrete_Data.xls','5_BETAPLASMA.csv', '6_Drybulbtemperature.xlsx', '7_moisture content of raw material.xlsx', 
   '8_steam pressure.xlsx', '9_main stem temperature.xlsx','10_reheat steam temperature.xlsx']

   targets = ['y', 'y', 'Concrete compressive strength(MPa, megapascals) ','BETAPLASMA','dry bulb temperature',
   'moisture content of raw material','steam pressure','main stem temperature','reheat steam temperature']


   for index, (dataset_name, target) in enumerate(zip(dataset_names, targets)):
      temp = []
      for i in range(5):
         No_PCGrad = UpperLowerBound(dataset_name, target)
         PCGrad = UpperLowerBound(dataset_name, target)
         
         obj = Run(No_PCGrad, PCGrad)
         obj.run_no_pcgrad()
         obj.run_pcgrad()
         res = obj.print_comparison()
         temp.append(res)
      output = pd.concat(temp)
      output.to_csv(f'output_{index+1}.csv')
 
if __name__ == "__main__":
   # set the epochs 
   Run.epochs = 5
   # set the batch_size 
   # Run.batch_size = 1000
   main()















