import numpy as np
import pandas as pd
from pandas import DataFrame


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to apply all feature engineering steps.

    Only row-local and backward-looking steps run here, so it is safe on the full
    dataframe before the split. Frequency encoding and imputation learn from data,
    so they run after the split, fitted on train only.

    Args:
        df (pd.DataFrame): The raw input dataframe.

    Returns:
        pd.DataFrame: The engineered dataframe.
    """
    df = df.copy()
    df = add_time_features(df)
    df = add_missingness_flags(df)
    df = add_behavioral_aggregates(df)

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
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
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
    for col in COLS_TO_FLAG:
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


def _window_stats(
    times: np.ndarray, amounts: np.ndarray, window_secs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Summarise each transaction's trailing time window, for one card's sorted history.

    Args:
        times (np.ndarray): Transaction times in seconds, ascending.
        amounts (np.ndarray): Transaction amounts, which may contain NaN.
        window_secs (int): The length of the window.

    Returns:
        tuple: The transaction count in the window, including the current one, and the
        mean amount of the earlier transactions in it (NaN if there are none).
    """
    positions = np.arange(len(times))
    starts = np.searchsorted(times, times - window_secs, side="left")
    known = ~np.isnan(amounts)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, amounts, 0.0))))
    seen = np.concatenate(([0], np.cumsum(known)))

    earlier = seen[positions] - seen[starts]
    means = np.full(len(times), np.nan)
    has_earlier = earlier > 0
    means[has_earlier] = (total[positions] - total[starts])[has_earlier] / earlier[
        has_earlier
    ]
    return positions - starts + 1, means


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
    times = df["TransactionDT"].to_numpy(dtype=float)
    amounts = df["TransactionAmt"].to_numpy(dtype=float)
    card_rows = df.groupby("card1").indices.values()

    for window_secs, label in [(3600, "1h"), (86400, "24h")]:
        counts = np.full(len(df), np.nan)
        means = np.full(len(df), np.nan)
        for rows in card_rows:
            counts[rows], means[rows] = _window_stats(
                times[rows], amounts[rows], window_secs
            )
        df[f"card1_count_{label}"] = counts
        df[f"card1_mean_amt_{label}"] = means

    df["amt_deviation"] = df["TransactionAmt"] - df["card1_mean_amt_24h"]

    # Seconds since the same card's previous transaction; NaN for its first one.
    gaps = np.full(len(df), np.nan)
    for rows in card_rows:
        gaps[rows[1:]] = np.diff(times[rows])
    df["card1_secs_since_prev"] = gaps

    return df


# Numeric medians learned from train by impute_missing(fit=True).
num_medians: dict = {}


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
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    df[cat_cols] = df[cat_cols].fillna("missing")

    if fit:
        num_medians.clear()
        # One column at a time in float64: the medians a median SimpleImputer
        # gives, without copying the whole frame to float64 (1.5 GB on train).
        for col in num_cols:
            median = df[col].astype("float64").median()
            num_medians[col] = 0.0 if pd.isna(median) else float(median)

    df.fillna({col: num_medians.get(col, 0.0) for col in num_cols}, inplace=True)

    return df


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def prepare_transaction(tx: dict, prep: dict, feature_cols: list[str]) -> dict:
    """
    Turn one raw transaction into model inputs the way training did.

    A model column the caller sends as a number is kept as is. Otherwise the
    value is derived from raw fields (time features, missingness flags,
    frequency and category codes, using `prep` from inference_artifacts) or
    filled with the training median. History features such as card counts need
    the card's past transactions, so a caller without them gets the median.
    """
    raw = {k: v for k, v in tx.items() if v is not None}
    derived: dict = {}

    dt = raw.get("TransactionDT")
    if _is_number(dt):
        hour = (dt // 3600) % 24
        derived.update(
            tx_hour=hour,
            tx_day=(dt // 86400) % 7,
            tx_hour_sin=np.sin(2 * np.pi * hour / 24),
            tx_hour_cos=np.cos(2 * np.pi * hour / 24),
        )
    for col in COLS_TO_FLAG:
        derived[f"{col}_was_missing"] = int(col not in raw)
    for col, freqs in prep.get("freq_maps", {}).items():
        value = raw.get(col)
        hashable = isinstance(value, (str, int, float))
        derived[f"{col}_freq"] = freqs.get(value, 0.0) if hashable else 0.0

    codes_by_col = prep.get("category_codes", {})
    medians = prep.get("medians", {})
    row = {}
    for col in feature_cols:
        value = raw.get(col)
        if _is_number(value):
            row[col] = value
        elif col in codes_by_col:
            codes = codes_by_col[col]
            key = "missing" if value is None else str(value)
            row[col] = codes.get(key, codes.get("__unknown__", 0))
        elif col in derived:
            row[col] = derived[col]
        else:
            row[col] = medians.get(col, 0.0)
    return row
