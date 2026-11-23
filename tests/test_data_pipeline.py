import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_behavioral_aggregates,
    add_frequency_encoding,
    add_missingness_flags,
    add_time_features,
    engineer_features,
    impute_missing,
    prepare_transaction,
)
from src.data.preprocessing import (
    apply_smote,
    encode_categoricals,
    fit_transform_split,
    inference_artifacts,
)


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

    result = impute_missing(add_frequency_encoding(engineer_features(frame)))

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


def test_later_rows_never_change_earlier_rows_features() -> None:
    early = pd.DataFrame(
        {
            "TransactionDT": [100, 7200, 90000],
            "TransactionAmt": [10.0, None, 30.0],
            "card1": [1, 1, 2],
            "card4": ["visa", None, "visa"],
            "id_01": [None, 1.0, 2.0],
        }
    )
    late = pd.DataFrame(
        {
            "TransactionDT": [900000],
            "TransactionAmt": [9999.0],
            "card1": [3],
            "card4": ["master"],
            "id_01": [500.0],
        }
    )

    alone = engineer_features(early)
    with_future = engineer_features(pd.concat([early, late], ignore_index=True)).iloc[
        :3
    ]

    pd.testing.assert_frame_equal(alone, with_future)


def test_smote_keeps_the_real_rows_first() -> None:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    labels = pd.Series([1] * 10 + [0] * 190)

    X_res, Y_res = apply_smote(features, labels, sampling_strategy=0.5)

    assert (X_res.iloc[:200].to_numpy() == features.to_numpy()).all()
    assert (Y_res.iloc[200:] == 1).all()


def test_gap_since_previous_transaction_looks_only_at_the_same_card() -> None:
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 160, 400, 1000],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": [1, 2, 1, 1],
        }
    )

    out = add_behavioral_aggregates(df)

    gaps = out.sort_values("TransactionDT")["card1_secs_since_prev"].tolist()
    assert np.isnan(gaps[0]) and np.isnan(gaps[1])
    assert gaps[2:] == [300.0, 600.0]


def test_one_raw_transaction_is_prepared_exactly_like_the_batch_pipeline() -> None:
    raw = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionDT": [100, 7200, 90000, 180000, 200000],
            "TransactionAmt": [10.0, 20.0, None, 40.0, 55.0],
            "card1": [1.0, 1.0, 2.0, 2.0, 3.0],
            "card4": ["visa", "visa", None, "master", "amex"],
            "id_01": [None, 1.0, 2.0, None, None],
        }
    )
    engineered = engineer_features(raw)
    train, test = engineered.iloc[:3].copy(), engineered.iloc[3:].copy()
    raw_test = test.copy()  # what a caller would send, before fitting

    _, prepared = fit_transform_split(train, test)
    cols = [c for c in prepared.columns if c not in ("TransactionID", "isFraud")]
    prep = inference_artifacts()

    for i in range(len(raw_test)):
        sent = {
            k: (None if pd.isna(v) else v)
            for k, v in raw_test.iloc[i].to_dict().items()
            if k != "isFraud"
        }
        one = prepare_transaction(sent, prep, cols)
        assert one == pytest.approx(prepared.iloc[i][cols].to_dict())


def test_a_bare_transaction_gets_training_medians_and_unknown_codes() -> None:
    prep = {
        "medians": {"C1": 3.0},
        "freq_maps": {"P_emaildomain": {"gmail.com": 0.4}},
        "category_codes": {"card4": {"missing": 0, "visa": 1, "__unknown__": 2}},
    }
    cols = ["C1", "P_emaildomain_freq", "card4", "id_01_was_missing", "TransactionAmt"]

    one = prepare_transaction(
        {"TransactionAmt": 25, "P_emaildomain": "gmail.com", "card4": "bitcoin"},
        prep,
        cols,
    )

    assert one == {
        "C1": 3.0,
        "P_emaildomain_freq": 0.4,
        "card4": 2,
        "id_01_was_missing": 1,
        "TransactionAmt": 25,
    }
