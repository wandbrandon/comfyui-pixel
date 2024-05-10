import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Generate sample data: points on a circle
num_points = 16
np.random.seed(0)
theta = np.random.uniform(-np.pi, np.pi, size=num_points)
x = np.cos(theta)
y = np.sin(theta)
points = np.column_stack((x, y))

plt.figure()
plt.scatter(x, y)
plt.show()

# DBSCAN
clustering = DBSCAN(eps=0.1, min_samples=1, metric="cosine", n_jobs=-1).fit(points)
labels = clustering.labels_

plt.figure()
plt.scatter(x, y, c=labels, s=200, cmap="viridis")
plt.show()

# Number of clusters found
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Labels:", labels)
print("Number of clusters:", num_clusters)
