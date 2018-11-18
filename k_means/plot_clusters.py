import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

MAX_K = 7
BASE_PATH = 'clustered/{}.txt'

for i in range(1, MAX_K+1):
	path = BASE_PATH.format(i)	
	colors = plt.get_cmap('jet')(np.linspace(0, 1.0, i))
	
	with open(path) as file:
		lines = file.readlines()
	
	centroids = {}
	for line in lines[:i]:
		n, x1, x2 = line.split('\t')
		centroids[int(n)] = float(x1), float(x2)
		
	lines = lines[i:]
	clusters = {n: [[],[]] for n in centroids}
	
	for line in lines:
		n, distance, speed = line.split('\t')
		clusters[int(n)][0].append(float(distance))
		clusters[int(n)][1].append(float(speed))
		
	plt.figure(i, figsize=(9,7))
	
	for key, cluster in clusters.items():
		plt.scatter(cluster[0], cluster[1], c=[colors[key]], label=f"Cluster {key}")
		plt.plot([centroids[key][0]], [centroids[key][1]], 'X', 
				 c=colors[key], markersize=12, mew=2, mec='black')
																			   
	plt.xlim((0, 260))
	plt.ylim((-5, 105))
	plt.legend(loc='best')
	
	plt.xlabel('Distance feature')
	plt.ylabel('Speed feature')
plt.show()
