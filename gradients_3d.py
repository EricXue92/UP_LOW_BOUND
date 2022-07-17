import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
sns.set_theme(style="darkgrid")

# Plot the ditribution 
def plot_pred_distribution(file_name = 'no_pcgrad_pred.csv', x = 'Width'):
	df = pd.read_csv(file_name)
	sns.displot(data=df, x = x, kde=True)
	sns.set_style("ticks")
	plt.title(x)
	plt.savefig(f'{file_name}_{x}_Distribution.png')
	plt.show()

# val_acc is on the validation data, val_loss is a good indication of how the model performs on unseen data.
def plot_val_loss_3D(file_name = 'history_no_pcgrad_history.pkl'):
	dict_data = pd.read_pickle('history_no_pcgrad_history.pkl')  
	df = pd.DataFrame(dict_data)
	width, coverage = df['val_mpiw'], df['val_coverage']
	val_loss = df['val_loss']
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111, projection='3d')
	#ax.plot(width, coverage, z)
	#ax.scatter(width, coverage, z)
	#ax.plot_trisurf(width, coverage, z, linewidth=0, antialiased=True)
	#surf = ax.plot_trisurf(width-np.mean(width), coverage-np.mean(coverage), val_loss, cmap=cm.jet, linewidth=0)
	surf = ax.plot_trisurf(width, coverage, val_loss, cmap=cm.jet, linewidth=0)
	fig.colorbar(surf)
	ax.set_xlabel('val_width')
	ax.set_ylabel('val_coverage')
	ax.set_title('val_Loss')
	ax.xaxis.set_major_locator(MaxNLocator(10))
	ax.yaxis.set_major_locator(MaxNLocator(10))
	ax.zaxis.set_major_locator(MaxNLocator(10))
	fig.tight_layout()
	plt.savefig(f'{file_name}_3D.png', dpi = 600)
	plt.show()

# To show the same y_slim as history_no_pcgrad_histor
def plot_pcgrad_history_with_same_yslim():
	dict_data = pd.read_pickle('history_pcgrad_history.pkl')
	df = pd.DataFrame(dict_data)
	plt.ylim(0, 1.4)
	fig = plt.figure(figsize=(10,6))
	sns.set_style("ticks")
	plt.xlabel("Epochs")
	ax = sns.lineplot(data=df[ ['coverage', 'mpiw', 'coverage_width_rate', 'val_coverage', 'val_mpiw', 'val_loss', 'val_coverage_width_rate']])
	ax.set(ylim=(0, 1.4))
	plt.savefig(f'history_pcgrad_ylim_png', dpi = 600)
	plt.show()

def plot_Coverage_rate_Width_rate(file_name = 'no_pcgrad_pred.csv'):
	df = pd.read_csv(file_name)
	x = np.array(range(0,len(df),1))
	y = df['Coverage_rate/Width']
	plt.plot(x, y)
	plt.savefig(f'{file_name}_coverage_rate_width.png')
	plt.show()

def main():
	# #Plot prediction distribution
	# plot_pred_distribution()
	# plot_pred_distribution(x = 'Coverage_rate')
	# plot_pred_distribution(file_name = 'pcgrad_pred.csv')
	# plot_pred_distribution(file_name = 'pcgrad_pred.csv', x ='Coverage_rate')

	# #Plot the loss_val, val_mpiw, val_coverage
	# plot_val_loss_3D()
	# plot_val_loss_3D('history_pcgrad_history.pkl')

	# #Plot the same y_slim as history_no_pcgrad_history
	# plot_pcgrad_history_with_same_yslim()

	# #plot Coverage_rate/Width
	plot_Coverage_rate_Width_rate()
	plot_Coverage_rate_Width_rate('pcgrad_pred.csv')


if __name__ == "__main__":
   main()




