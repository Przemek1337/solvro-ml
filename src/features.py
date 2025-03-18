import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def ExtractIngredientMatrix(df: pd.DataFrame) -> pd.DataFrame:

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
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return scaled_features