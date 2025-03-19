import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP
from typing import Tuple


def PlotClusters2D(features: pd.DataFrame, labels: np.ndarray, title: str = "Cluster Visualization"):
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()


def PlotTSNE(features: pd.DataFrame, labels: np.ndarray):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", alpha=0.7)
    plt.title("t-SNE Cluster Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()


# def PlotUMAP(features: pd.DataFrame, labels: np.ndarray):
#     reducer = umap(n_components=2, random_state=42)
#     reduced_features = reducer.fit_transform(features)
#
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", alpha=0.7)
#     plt.title("UMAP Cluster Visualization")
#     plt.xlabel("UMAP Component 1")
#     plt.ylabel("UMAP Component 2")
#     plt.legend(title="Cluster")
#     plt.show()


def PlotClusterDistribution(labels: np.ndarray):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels, hue=labels, palette="viridis", legend=False)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()