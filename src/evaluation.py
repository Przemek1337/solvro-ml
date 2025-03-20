import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Any

def EvaluateClusters(features: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering results using internal metrics.

    Args:
        features: Scaled feature matrix
        labels: Cluster labels

    Returns:
        Dict: Dictionary with evaluation metrics
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(features) - 1:
        return {'error': 'Invalid number of clusters for evaluation'}

    metrics = {
        'silhouette_score': silhouette_score(features, labels),
        'davies_bouldin_score': davies_bouldin_score(features, labels),
        'calinski_harabasz_score': calinski_harabasz_score(features, labels)
    }

    return metrics

def AnalyzeClusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Analyze the composition of clusters.

    Args:
        df: Original cocktail DataFrame
        labels: Cluster labels

    Returns:
        pd.DataFrame: Summary of cluster characteristics
    """
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    clusters = np.unique(labels)
    cluster_summary = []

    for cluster in clusters:
        if cluster == -1:
            continue

        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster]
        cluster_info = {
            'cluster': cluster,
            'size': len(cluster_df)
        }
        if 'category' in cluster_df.columns:
            cluster_info['top_categories'] = cluster_df['category'].value_counts().head(3).to_dict()

        if 'glass' in cluster_df.columns:
            cluster_info['top_glasses'] = cluster_df['glass'].value_counts().head(3).to_dict()

        if 'ingredients' in cluster_df.columns:
            ingredient_counts = {}
            for ingredients in cluster_df['ingredients']:
                if isinstance(ingredients, list):
                    for ingredient in ingredients:
                        if isinstance(ingredient, dict) and 'name' in ingredient:
                            name = ingredient['name'].lower()
                            ingredient_counts[name] = ingredient_counts.get(name, 0) + 1

            cluster_info['top_ingredients'] = {k: v for k, v in sorted(
                ingredient_counts.items(), key=lambda item: item[1], reverse=True)[:5]}

        if 'name' in cluster_df.columns:
            cluster_info['sample_cocktails'] = cluster_df['name'].sample(min(5, len(cluster_df))).tolist()

        cluster_summary.append(cluster_info)

    return pd.DataFrame(cluster_summary)



