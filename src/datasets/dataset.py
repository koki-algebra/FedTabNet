from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_dataset(
    file_path: str, target: str, labeled_size: float, train_size=0.8
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[int], List[int]]:
    df: pd.DataFrame = pd.read_csv(file_path, sep=",")

    train_l_df: pd.DataFrame
    train_u_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = train_test_split(df, train_size=train_size)
    train_l_df, train_u_df = train_test_split(train_df, train_size=labeled_size)

    train_l_indices = train_l_df.index.values
    train_u_indices = train_u_df.index.values
    test_indices = test_df.index.values

    # Simple preprocessing
    # label encode categorical features and fill empty cells
    nunique = df.nunique()
    types = df.dtypes

    categorical_columns = []
    categorical_dims = {}

    for col in df.columns:
        if types[col] == "object" or nunique[col] < 200:
            l_enc = LabelEncoder()
            df[col] = df[col].fillna("VV_likely")
            df[col] = l_enc.fit_transform(df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            df[col].fillna(df.loc[train_l_indices, col].mean(), inplace=True)

    # Define categorical features for categorical embeddings
    features = [col for col in df.columns if col != target]

    # indices of categorical features
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    # dimensions of categorical features
    cat_dims = [
        categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns
    ]

    # train labeled
    X_l_train = df[features].values[train_l_indices]
    y_l_train = df[target].values[train_l_indices]

    # train unlabeled
    X_u_train = df[features].values[train_u_indices]
    y_u_train = df[target].values[train_u_indices]

    # test
    X_test = df[features].values[test_indices]
    y_test = df[target].values[test_indices]

    dataset = {
        "train_labeled": (X_l_train, y_l_train),
        "train_unlabeled": (X_u_train, y_u_train),
        "test": (X_test, y_test),
    }

    return dataset, cat_idxs, cat_dims
