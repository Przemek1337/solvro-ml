import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Optional
import numpy as np
def LoadDataset(file_path: str) -> pd.DataFrame:
    """
        Load cocktail data from JSON file.

        Args:
            file_path: Path to the JSON file with cocktail data

        Returns:
            pd.DataFrame: DataFrame containing cocktail data
    """
    return pd.read_json(file_path)

def CleanData(df: pd.DataFrame) -> pd.DataFrame:
    """
        Clean the cocktail dataset by handling missing values and inconsistencies.

        Args:
            df: Raw cocktail DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    return cleaned_df

def ExtractIngriedientsFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the ingredients column.

    Args:
        df: DataFrame with cocktail data

    Returns:
        pd.DataFrame: DataFrame with extracted ingredient features
    """
    all_ingredient_names = set()
    for ingredients_list in df['ingredients']:
        for ingredient in ingredients_list:
            all_ingredient_names.add(ingredient['name'])
    ingredient_features = {f'has_{name}': [] for name in all_ingredient_names}

    for _, row in df.iterrows():
        current_ingredients = {ingredient['name'] for ingredient in row['ingredients']}
        for name in all_ingredient_names:
            ingredient_features[f'has_{name}'].append(1 if name in current_ingredients else 0)

    ingredients_df = pd.DataFrame(ingredient_features)

    result_df = pd.concat([df.drop(columns=['ingredients']), ingredients_df], axis=1)

    return result_df