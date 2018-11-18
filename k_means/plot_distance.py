import matplotlib.pyplot as plt
import numpy as np

with open("average_distance_by_k.txt") as file:
	lines = file.readlines()
	
k_distances = {}
for line in lines:
	k, distance = line.split('\t')
	
	k_distances[int(k)] = float(distance)
	
plt.bar(k_distances.keys(), k_distances.values(), width=0.9, color='red')

plt.xticks(list(k_distances.keys()), list(k_distances.keys()))
plt.yticks(np.arange(0, 51, 5))

plt.xlabel('Number of clusters (k)')
plt.ylabel('Average intracluster distance')

plt.show()
