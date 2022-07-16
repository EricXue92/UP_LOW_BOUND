import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# def plot_data(file_name = 'no_pcgrad_pred.csv', x = 'Width'):

# 	df = pd.read_csv(file_name)
# 	sns.displot(data=df, x = x, kde=True)
# 	sns.set_style("ticks")
# 	plt.title(x)
# 	plt.savefig(f'{file_name}_{x}_Distribution.png')
# 	plt.show()

# plot_data()
# plot_data(x = 'Coverage_rate')
# plot_data(file_name = 'pcgrad_pred.csv')
# plot_data(file_name = 'pcgrad_pred.csv', x ='Coverage_rate')
 

def total_gradients(width , coverage):
	return np.square(width) + coverage

def plot_3d(file_name = 'history_no_pcgrad_history.pkl'):
	dict_data = pd.read_pickle('history_no_pcgrad_history.pkl')  
	df = pd.DataFrame(dict_data)
	width, coverage = df['mpiw'], df['coverage']
	z = total_gradients(width, coverage)
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111, projection='3d')

	#ax.plot(width, coverage, z)
	#ax.scatter(width, coverage, z)

	#ax.plot_trisurf(width, coverage, z, linewidth=0, antialiased=True)
	#surf = ax.plot_trisurf(width-np.mean(width), coverage-np.mean(coverage), z, cmap=cm.jet, linewidth=0)
	surf = ax.plot_trisurf(width, coverage, z, cmap=cm.jet, linewidth=0)
	fig.colorbar(surf)
	ax.set_title('Grediant')
	plt.savefig(f'{file_name}_3D.png', dpi = 600)
	ax.xaxis.set_major_locator(MaxNLocator(14))
	ax.yaxis.set_major_locator(MaxNLocator(10))
	ax.zaxis.set_major_locator(MaxNLocator(10))
	fig.tight_layout()
	plt.show()

# plot_3d()
plot_3d('history_pcgrad_history.pkl')


# # To show the same y_slim as history_no_pcgrad_histor
# dict_data = pd.read_pickle('history_pcgrad_history.pkl')
# df = pd.DataFrame(dict_data)
# plt.ylim(0, 1.4)
# fig = plt.figure(figsize=(10,6))
# sns.set_style("ticks")
# plt.xlabel("Epochs")
# ax = sns.lineplot(data=df[ ['coverage', 'mpiw', 'val_coverage', 'val_mpiw']])
# ax.set(ylim=(0, 1.4))
# plt.savefig(f'history_pcgrad_ylim_png', dpi = 600)
# plt.show()





