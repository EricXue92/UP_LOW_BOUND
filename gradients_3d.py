import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv('no_pcgrad_pred.csv')

width = np.array(df['Width'])
coverage = np.array(df['Coverage_rate'])

def total_gradients(width, coverage):
	return np.square(width) + coverage

z = total_gradients(width, coverage)

fig = plt.figure(figsize=(10,6))

ax = plt.axes(projection='3d')

ax.plot_trisurf(width, coverage, z, linewidth=0, antialiased=False)

# ax.plot_surface(width, coverage, z, rstride = 1, 
# 					cstride =1 , cmap ='viridis', 
# 					edgecolor = 'none')
ax.set_title('surface')
plt.savefig('3D.png')
plt.show()