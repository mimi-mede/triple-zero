import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN 

pdf = pd.read_csv("C:\\coffee\\house_sales.csv")
pdf = pdf[['lat','long']]
pdf.drop_duplicates()
pdf.head(5)

msk = np.random.rand(len(pdf)) < 0.5
pdf = pdf[msk]
plt.scatter(pdf.lat,pdf.long, color = 'black', marker='.')
plt.show()

epsilon = 0.03
minimumSamples = 10
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(pdf)
labels = db.labels_
labels


# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Plot the points with colors
for k, col in zip(unique_labels, colors):
  print(k)
  class_member_mask = (labels == k)

  # Plot the datapoints that are clustered
  xy = pdf[class_member_mask & core_samples_mask]
  plt.scatter(xy.lat, xy.long,s=50, c=[col], marker=u'.', alpha=0.5)

  # Plot the outliers
  xy = pdf[class_member_mask & ~core_samples_mask]
  plt.scatter(xy.lat, xy.long,s=50, c='black', marker=u'o', alpha=0.5)
plt.show()

