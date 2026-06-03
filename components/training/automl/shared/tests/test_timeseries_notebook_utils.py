"""Tests for time series inference notebook helpers."""

from __future__ import annotations

import pandas as pd

from ..timeseries_notebook_utils import fill_known_covariates_on_future_frame


def _future_frame(item_ids: list, timestamps: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(item_id, pd.Timestamp(ts)) for item_id in item_ids for ts in timestamps],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame(index=index)


class TestFillKnownCovariatesOnFutureFrame:
    """Tests for known-covariate filling on the forecast horizon."""

    def test_uses_exact_sample_values_for_matching_timestamps(self):
        """Covariates match sample rows when the forecast timestamp is in the sample."""
        score_df = pd.DataFrame(
            {
                "item_id": [1, 1, 1],
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "promo": [0.0, 1.0, 0.5],
            }
        )
        future_cov = _future_frame([1], ["2025-01-03", "2025-01-04"])

        result = fill_known_covariates_on_future_frame(
            future_cov,
            score_df,
            "item_id",
            "timestamp",
            ["promo"],
        )

        assert result["promo"].tolist() == [0.5, 0.5]

    def test_fills_future_steps_with_last_known_value_per_item(self):
        """Future horizon steps use the last known covariate value per item."""
        score_df = pd.DataFrame(
            {
                "item_id": [1, 1],
                "timestamp": ["2025-01-01", "2025-01-02"],
                "promo": [0.0, 1.0],
            }
        )
        future_cov = _future_frame([1], ["2025-01-03", "2025-01-04"])

        result = fill_known_covariates_on_future_frame(
            future_cov,
            score_df,
            "item_id",
            "timestamp",
            ["promo"],
        )

        assert result["promo"].tolist() == [1.0, 1.0]

    def test_aligns_object_item_ids_to_future_index_dtype(self):
        """Regression for int64 future index vs object sample ids (reset_index merge bug)."""
        score_df = pd.DataFrame(
            {
                "item_id": ["1", "1"],
                "timestamp": ["2025-01-01", "2025-01-02"],
                "promo": [0.0, 1.0],
            }
        )
        future_cov = _future_frame([1], ["2025-01-03"])

        result = fill_known_covariates_on_future_frame(
            future_cov,
            score_df,
            "item_id",
            "timestamp",
            ["promo"],
        )

        assert result["promo"].iloc[0] == 1.0

    def test_multiple_items_get_item_specific_last_values(self):
        """Each item receives its own last-known covariate value."""
        score_df = pd.DataFrame(
            {
                "item_id": [1, 1, 2, 2],
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
                "promo": [0.0, 1.0, 10.0, 20.0],
            }
        )
        future_cov = _future_frame([1, 2], ["2025-01-03"])

        result = fill_known_covariates_on_future_frame(
            future_cov,
            score_df,
            "item_id",
            "timestamp",
            ["promo"],
        )

        promo_by_item = result.reset_index().set_index("item_id")["promo"]
        assert promo_by_item.loc[1] == 1.0
        assert promo_by_item.loc[2] == 20.0
