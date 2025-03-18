from typing import Tuple, Any, Dict

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
def KMEANS(features: pd.DataFrame, n_clusters: int = 5) -> Tuple[np.ndarray, Any]:
    """
       Apply K-means clustering to feature matrix.

       Args:
           features: Scaled feature matrix
           n_clusters: Number of clusters

       Returns:
           Tuple: (cluster_labels, model)
       """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit(features)

    return labels, kmeans

def AC(features: pd.DataFrame, n_clusters: int = 5) -> Tuple[np.ndarray, Any]:
    """
       Apply hierarchical clustering to feature matrix.

       Args:
           features: Scaled feature matrix
           n_clusters: Number of clusters

       Returns:
           Tuple: (cluster_labels, model)
       """
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(features)
    return labels, ac
def DBSCAN(features: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Any]:
    """
       Apply DBSCAN clustering to feature matrix.

       Args:
           features: Scaled feature matrix
           eps: Maximum distance between samples
           min_samples: Minimum number of samples in a neighborhood

       Returns:
           Tuple: (cluster_labels, model)
       """
    ds = DBSCAN(eps=eps, min_samples=min_samples)
    labels = ds.fit_predict(features)
    return labels, ds
def FindOptimalK(features: pd.DataFrame, max_k: int = 15) -> int:
    """
       Find optimal number of clusters using the elbow method.

       Args:
           features: Scaled feature matrix
           max_k: Maximum number of clusters to try

       Returns:
           int: Optimal number of clusters
       """
    silhoette_scores = []
    inertia_values = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        silhoette_scores.append(silhouette_score(features, labels))
        inertia_values.append(kmeans.inertia_)

    optimal_k = np.argmax(silhoette_scores) + 2

    return optimal_k

def ReduceDimensions(features: pd.DataFrame, method: str = 'pca', n_components: int = 2) -> np.ndarray:
    """
        Reduce dimensions for visualization.

        Args:
            features: Scaled feature matrix
            method: Dimension reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components for the output

        Returns:
            np.ndarray: Reduced features
        """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_features = reducer.fit_transform(features)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced_features = reducer.fit_transform(features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_features = reducer.fit_transform(features)
    else:
        raise ValueError('Invalid dimension reduction method')

    return reduced_features


