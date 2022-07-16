import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 

def plot_data(file_name = 'no_pcgrad_pred.csv', x = 'Width'):

	df = pd.read_csv(file_name)
	sns.displot(data=df, x = x, kde=True)
	sns.set_style("ticks")
	plt.title(x)
	plt.savefig(f'{file_name}_{x}_Distribution.png')
	plt.show()

plot_data()
plot_data(x = 'Coverage_rate')
plot_data(file_name = 'pcgrad_pred.csv')
plot_data(file_name = 'pcgrad_pred.csv', x ='Coverage_rate')
 

def total_gradients(width , coverage):
	return np.square(width) + coverage

def plot_3d(file_name = 'pcgrad_pred.csv'):
	df = pd.read_csv(file_name)
	width, coverage = df['Width'], df['Coverage_rate']
	z = total_gradients(width, coverage)
	fig = plt.figure(figsize=(10,6))
	ax = plt.axes(projection='3d')
	ax.plot_trisurf(width, coverage, z, linewidth=0, antialiased=False)
	ax.set_title('Grediant')
	plt.savefig(f'{file_name}_3D.png', dpi = 600)
	plt.show()

plot_3d()
plot_3d('no_pcgrad_pred.csv')

