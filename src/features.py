import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def CreateFeatureMatrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature matrix from cocktail data suitable for clustering.

    Args:
        df: Cleaned cocktail DataFrame

    Returns:
        pd.DataFrame: Feature matrix for clustering
    """
    cat_features = pd.get_dummies(df[["category", "glass", "alcoholic"]], drop_first=True)
    numeric_cols = df.select_dtypes(include=["int", "float", "bool"]).columns
    df_features = df[numeric_cols].copy()
    return df_features
def ExtractIngredientMatrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary matrix of ingredients from ingredients column.

    Args:
        df: DataFrame with ingredients column

    Returns:
        pd.DataFrame: Binary matrix where rows are cocktails and columns are ingredients
    """

    all_ingredients = set()

    for ingredients in df['ingredients']:
        if isinstance(ingredients, list):
            for ingredient in ingredients:
                if isinstance(ingredient, dict) and 'name' in ingredient:
                    all_ingredients.add(ingredient['name'].lower())

    ingredient_matrix = pd.DataFrame(0, index=df.index, columns=list(all_ingredients))

    for idx, ingredients in df['ingredients'].items():
        if isinstance(ingredients, list):
            for ingredient in ingredients:
                if isinstance(ingredient, dict) and 'name' in ingredient:
                    ingredient_name = ingredient['name'].lower()
                    ingredient_matrix.loc[idx,ingredient_name] = 1
    return ingredient_matrix

def ScaleFeatures(features: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features using StandardScaler.

    Args:
        features: Feature matrix

    Returns:
        pd.DataFrame: Scaled feature matrix
    """
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return scaled_features