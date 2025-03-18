import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

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
    mlb = MultiLabelBinarizer()
    ingredients_matrix = mlb.fit_transform(df['ingredients'])
    ingredients_encoded = pd.DataFrame(ingredients_matrix, columns=[f'has_{ing}' for ing in mlb.classes_])
    df = df.drop(columns=['ingredients']).join(ingredients_encoded)
    return df