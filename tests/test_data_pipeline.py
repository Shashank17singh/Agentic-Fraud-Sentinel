import pandas as pd
import pytest

from src.data.features import (
    add_behavioral_aggregates,
    add_missingness_flags,
    add_time_features,
    engineer_features,
)
from src.data.preprocessing import encode_categoricals


def test_missingness_flags_mark_missing_values() -> None:
    frame = pd.DataFrame({"id_01": [1.0, None], "DeviceType": [None, "mobile"]})

    flagged = add_missingness_flags(frame)

    assert flagged["id_01_was_missing"].tolist() == [0, 1]
    assert flagged["DeviceType_was_missing"].tolist() == [1, 0]


def test_missingness_flags_only_cover_known_columns() -> None:
    flagged = add_missingness_flags(pd.DataFrame({"other": [None, 2.0]}))

    assert "other_was_missing" not in flagged


def test_unseen_categories_get_their_own_code() -> None:
    train = pd.DataFrame({"card4": ["visa", "master", "visa"], "isFraud": [0, 1, 0]})
    test = pd.DataFrame({"card4": ["visa", "amex"], "isFraud": [0, 0]})

    encoded_train, encoded_test = encode_categoricals(train, test)[:2]

    assert encoded_test["card4"].iloc[0] == encoded_train["card4"].iloc[0]
    assert encoded_test["card4"].iloc[1] not in set(encoded_train["card4"])


def test_time_features_are_derived_from_transaction_time() -> None:
    frame = pd.DataFrame({"TransactionDT": [0, 3600 * 6]})

    result = add_time_features(frame)

    assert result["tx_hour"].tolist() == [0, 6]
    assert result["tx_hour_sin"].iloc[1] == pytest.approx(1)


def test_the_full_feature_pipeline_runs_and_leaves_no_gaps() -> None:
    frame = pd.DataFrame(
        {
            "TransactionDT": [100, 7200, 90000, 180000],
            "TransactionAmt": [10.0, 20.0, None, 40.0],
            "card1": [1, 1, 2, 2],
            "card4": ["visa", "visa", None, "master"],
            "id_01": [None, 1.0, 2.0, None],
        }
    )

    result = engineer_features(frame)

    assert not result.isnull().any().any()
    for column in ("tx_hour", "id_01_was_missing", "card4_freq", "amt_deviation"):
        assert column in result


def test_windows_only_count_recent_transactions_for_the_same_card() -> None:
    hour = 3600
    frame = pd.DataFrame(
        {
            "TransactionDT": [0, 2 * hour, 2 * hour + 60, 30 * hour, 60],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 500.0],
            "card1": [1, 1, 1, 1, 2],
        }
    )

    result = add_behavioral_aggregates(frame)
    card = result[result["card1"] == 1].reset_index(drop=True)

    assert card["card1_count_1h"].tolist() == [1, 1, 2, 1]
    assert card["card1_count_24h"].tolist() == [1, 2, 3, 1]
    assert card["card1_mean_amt_24h"].iloc[2] == pytest.approx(15.0)
    assert card["card1_mean_amt_1h"].iloc[2] == pytest.approx(20.0)
    assert pd.isna(card["card1_mean_amt_1h"].iloc[0])


def test_the_current_amount_is_never_included_in_its_own_mean() -> None:
    frame = pd.DataFrame(
        {"TransactionDT": [0, 10], "TransactionAmt": [10.0, 1000.0], "card1": [1, 1]}
    )

    assert add_behavioral_aggregates(frame)["card1_mean_amt_1h"].iloc[1] == 10.0

