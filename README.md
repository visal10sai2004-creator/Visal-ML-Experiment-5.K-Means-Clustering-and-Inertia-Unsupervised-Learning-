##  Experiment-5.K-Means Clustering and Inertia (Unsupervised Learning)

## AIM:

Apply KMeans to a synthetic 2-cluster dataset; compute and visualize inertia.

## ALGORITHM:
 
 Lloyd’s algorithm: assign to nearest centroid → recompute centroids until convergence.

## THEORY:

KMeans minimizes within-cluster variance; relates to vector quantization.

## PROGRAM:

```

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
np.random.seed(42)
X = np.vstack([np.random.randn(50,2)+np.array([0,0]), 
np.random.randn(50,2)+np.array([5,5])])
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis'); 
plt.scatter(centers[:,0], centers[:,1], marker='x', color='red', 
s=100); plt.title('KMeans Clusters'); plt.show()
print('Inertia (WCSS):', round(kmeans.inertia_,3))

```

## OUTPUT:

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/ef107b99-b980-4d7f-acbc-a0910ac36fd5" />

```

Inertia (WCSS): 170.203

```

## RESULT:

Students observe cluster structure and effect of centroid positions. Note sensitivity to initialization & scaling.
