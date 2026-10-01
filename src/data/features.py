import numpy as np
import pandas as pd
from pandas import DataFrame


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to apply all feature engineering steps.
    
    Call this on the full joined dataframe before splitting into train/test sets.
    
    Args:
        df (pd.DataFrame): The raw input dataframe.
        
    Returns:
        pd.DataFrame: The engineered dataframe.
    """
    df = df.copy()
    df = add_time_features(df)
    df = add_missingness_flags(df)
    df = add_frequency_encoding(df)
    df = add_behavioral_aggregates(df)
    df = impute_missing(df)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclic time signals from TransactionDT.
    
    TransactionDT is seconds since an arbitrary reference point.
    Extracting hour and day patterns helps identify fraud time trends.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'TransactionDT'.
        
    Returns:
        pd.DataFrame: Dataframe with cyclic time features added.
    """
    df["tx_day"] = (df["TransactionDT"] // 86400) % 7

    df["tx_hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24)
    df["tx_hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24)

    return df


COLS_TO_FLAG = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "DeviceType",
    "DeviceInfo",
]


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flags for missing values in key columns before imputing.
    
    Missing information (e.g., no device info attached) is often a strong fraud signal.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with missingness flags.
    """
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)

    return df


FREQ_COLS = [
    "card1",
    "card2",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
]

freq_maps: dict = {}


def add_frequency_encoding(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Replace each categorical value with its occurrence frequency.
    
    Rare values (e.g., unusual device, unknown email domain) will get low scores.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        fit (bool): If True, learns frequencies from the dataframe (use on train).
                    If False, applies stored frequencies (use on test).
                    
    Returns:
        pd.DataFrame: Dataframe with frequency-encoded columns.
    """
    for col in FREQ_COLS:
        if col not in df.columns:
            continue
        if fit:
            freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)
    return df


def add_behavioral_aggregates(df: pd.DataFrame) -> DataFrame:
    """
    Compute behavioral aggregate features for each card's recent history.
    
    Compares the current transaction to the card's history using rolling windows.
    Data is sorted by time first to ensure windows only look backward.
    
    Args:
        df (pd.DataFrame): Dataframe containing transaction amounts and times.
        
    Returns:
        pd.DataFrame: Dataframe with behavioral aggregates added.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    for window_secs, label in [(3600, "1h"), (86400, "24h")]:

        df[f"card1_count_{label}"] = df.groupby("card1")["TransactionDT"].transform(
            lambda x: x.expanding().count()
        )

        df[f"card1_mean_amt_{label}"] = df.groupby("card1")["TransactionAmt"].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    df["amt_deviation"] = df["TransactionAmt"] - df["card1_mean_amt_24h"]

    return df


from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")


def impute_missing(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Impute missing values for numerical and categorical columns.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        fit (bool): If True, fits imputers on this data (train only).
                    If False, transforms using already-fitted imputers (test).
                    
    Returns:
        pd.DataFrame: The imputed dataframe.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if fit:
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    else:
        df[num_cols] = num_imputer.transform(df[num_cols])
        df[cat_cols] = cat_imputer.transform(df[cat_cols])

    return df
