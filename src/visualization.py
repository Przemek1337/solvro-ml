import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def PlotClusters2D(features: pd.DataFrame, labels: np.ndarray, title: str = "Cluster Visualization"):
    """
         Plots 2D clusters using PCA for dimensionality reduction.

         Args:
             features (pd.DataFrame): The input data (features) to be visualized.
             labels (np.ndarray): An array of cluster labels for each row in the data.
             title (str, optional): The title of the plot. Defaults to "Cluster Visualization".

         Returns:
             None: This function displays a scatter plot and does not return any value.
     """
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
    """
        Plots 2D clusters using t-SNE for dimensionality reduction.

        Args:
            features (pd.DataFrame): The input data (features) to be visualized.
            labels (np.ndarray): An array of cluster labels for each row in the data.

        Returns:
            None: This function displays a scatter plot and does not return any value.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", alpha=0.7)
    plt.title("t-SNE Cluster Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()

def PlotClusterDistribution(labels: np.ndarray):
    """
        Plots a bar chart showing the distribution of cluster labels.

        Args:
            labels (np.ndarray): An array of cluster labels.

        Returns:
            None: This function displays a bar chart and does not return any value.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels, hue=labels, palette="viridis", legend=False)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()