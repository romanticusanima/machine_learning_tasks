import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, List, Optional

def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)

def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train and validation dataframes.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Exited"])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train and validation dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train and val sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> Any:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        numeric_cols (list): List of numerical columns.

    Returns:
        Scaler.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: Dict[str, Any], categorical_cols: list) -> Dict[str, Any]:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        categorical_cols (list): List of categorical columns.

    Returns:
        Dict[str, Any]: Dictionary containing encoder and encoded_cols.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    return {
        'encoder': encoder,
        'encoded_cols': encoded_cols,
    }


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = False) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.
    This function:
      - Drops rows with missing values in the 'Exited' column.
      - Drops unnecessary columns such as 'CustomerId' and 'Surname'.
      - Splits the data into training and validation sets.
      - Extracts numeric and categorical columns.
      - Optionally applies MinMax scaling to numeric features.
      - Applies one-hot encoding to categorical features.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        scaler_numeric (bool): Boolean.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - train_inputs, val_inputs: processed feature sets
            - train_targets, val_targets: target labels
            - input_cols: list of input feature columns
            - numeric_cols: list of numeric feature names
            - categorical_cols: list of categorical feature names
            - scaler (optional): fitted scaler if scaler_numeric is True
            - encoder: fitted OneHotEncoder
            - encoded_cols: names of encoded categorical columns
    """
    raw_df = drop_na_values(raw_df, ['Exited'])
    raw_df.drop(columns=['CustomerId', 'Surname'], inplace=True)

    split_dfs = split_data(raw_df)
    input_cols = list(raw_df.columns)[1:-1] # ignore Id, Exited
    target_col = 'Exited'
    
    data = create_inputs_targets(split_dfs, input_cols, target_col)
    data['input_cols'] = input_cols

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    data['numeric_cols'] = numeric_cols
    data['categorical_cols'] = categorical_cols

    if (scaler_numeric):
        data['scaler'] = scale_numeric_features(data, numeric_cols)
    else:
        data['scaler'] = None

    encoded_data = encode_categorical_features(data, categorical_cols)
    data['encoder'] = encoded_data['encoder']
    data['encoded_cols'] = encoded_data['encoded_cols']

    return data


def preprocess_new_data(input_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], encoded_cols: List[str], encoder: OneHotEncoder, scaler: Optional[MinMaxScaler]) -> pd.DataFrame:
    """
    Preprocess a new dataset using the fitted scaler and encoder.

    Args:
        input_df (pd.DataFrame): The raw dataframe.
        numeric_cols (List[str]): List of numeric feature column names.
        categorical_cols (List[str]): List of categorical feature column names.
        encoded_cols (List[str]): List of column names after one-hot encoding.
        encoder (OneHotEncoder): A fitted OneHotEncoder instance.
        scaler (Optional[MinMaxScaler]): A fitted MinMaxScaler instance or None.

    Returns:
        pd.DataFrame: A DataFrame containing the transformed numeric and encoded categorical features.
    """
    if (scaler):
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    
    return X_input