import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import glob 

# fig = plt.figure(figsize=(4, 4))
filenames = sorted(glob.glob('*.csv'))

def plot_box( to_plot = 'Coverage_rate'):

	for index, filename in enumerate(filenames):
		df = pd.read_csv(filename)
		No_PCGrad_df = df.iloc[::2]
	

		No_PCGrad_mean = round(No_PCGrad_df[to_plot].mean() * 100 , 2)

		PCGrad_df = df.iloc[1:len(df):2]
		
		PCGrad_mean = round(PCGrad_df[to_plot].mean() * 100, 2)

		No_PCGrad_df.reset_index(drop=True, inplace=True)
		PCGrad_df .reset_index(drop=True, inplace=True)
		df= pd.concat([No_PCGrad_df[to_plot], PCGrad_df[to_plot] ], ignore_index=True, axis = 1)
		df.columns = ['No_PCGrad','PCGrad']
		plt.title(f'Mean:{No_PCGrad_mean} %         Mean:{PCGrad_mean} %')

		ax = sns.boxplot(data = df)
		ax = sns.swarmplot(data = df, color=".25" )
		plt.savefig(f'{to_plot}_{index+1}.png')
		#plt.legend()
		plt.clf()

plot_box()
plot_box(to_plot = 'NMPIW')


