import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

## 'data' with 'embedding' column is generated from survey_emb{i}.py (i = 1, 2, or 3) ##

# PCA for embedding data
X = np.vstack(data["embedding"].values)
pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X)
print("PCA(50) explained variance ratio sum:", np.sum(pca_50.explained_variance_ratio_))

# PCA plot
plt.figure(figsize=(7, 6))
plt.scatter(X_pca_50[:, 0], X_pca_50[:, 1], c=data['score'], alpha=0.3)
plt.colorbar(label='score')
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA (colored by score)")
plt.tight_layout()
plt.savefig("pca_embedding.png", dpi=300, bbox_inches="tight")
plt.close()

# UMAP for PCA50
vectors_norm = normalize(X_pca_50, norm='l2')
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', n_components=2, random_state=42)
X_umap = umap_model.fit_transform(vectors_norm)

# UMAP plot
plt.figure(figsize=(7, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data['score'], alpha=0.2)
plt.colorbar(label="score")
plt.title("UMAP Projection colored by score")
plt.tight_layout()
plt.savefig("umap_pca50.png", dpi=300, bbox_inches="tight")
plt.close()

## clustering by DBSCAN ##
# parameters to be determined by user
ESP = 0.75 		 
Min_Sample = 30

dbscan = DBSCAN(eps=ESP, min_samples=Min_Sample) 
clusters = dbscan.fit_predict(X_umap)

## plot for clusters ##
plt.figure(figsize=(10, 8))

# label clusters by color
unique_clusters = sorted(set(clusters))
for cl in unique_clusters:
    mask = (clusters == cl)
    if cl == -1:    # in case of noise
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c='lightgrey', s=20, alpha=0.5,
            label='Noise (-1)'
        )
    else:    # in case of cluster
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            s=20, alpha=0.5,
            label=f'Cluster {cl}'
        )

plt.title(f"DBSCAN Clustering (eps={dbscan.eps}, min_samples={dbscan.min_samples})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("dbscan_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# count total clusters and noises
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
print(f"number of total clusters: {n_clusters_}")
print(f"number of noises: {n_noise_}")

# assign clusters to data
data['cluster'] = clusters

## save data to csv file ##
# data.to_csv("data.csv", index=False)
