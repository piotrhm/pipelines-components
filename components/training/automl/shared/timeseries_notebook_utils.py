"""Helpers for AutoGluon time series inference notebooks."""

from __future__ import annotations

import pandas as pd


def fill_known_covariates_on_future_frame(
    future_cov: pd.DataFrame,
    score_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    known_covariates_names: list[str],
) -> pd.DataFrame:
    """Fill known covariates on the forecast horizon from sample rows.

    Uses index alignment on (item_id, timestamp): exact matches from ``score_df``,
    then the last known covariate value per item for future steps not in the sample.

    Args:
        future_cov: Forecast-horizon frame from ``TimeSeriesPredictor.make_future_data_frame()``.
        score_df: Historical sample rows used for ``predict()``.
        id_column: Item/series id column in ``score_df``.
        timestamp_column: Timestamp column in ``score_df``.
        known_covariates_names: Covariate columns to populate on ``future_cov``.

    Returns:
        ``future_cov`` with ``known_covariates_names`` columns filled.
    """
    hist_cov = score_df[[id_column, timestamp_column, *known_covariates_names]].copy()
    hist_cov[timestamp_column] = pd.to_datetime(hist_cov[timestamp_column])
    hist_cov[id_column] = hist_cov[id_column].astype(future_cov.index.get_level_values(0).dtype)
    hist_cov = hist_cov.set_index([id_column, timestamp_column])[known_covariates_names]
    item_ids = future_cov.index.get_level_values(0)
    last_by_item = hist_cov.groupby(level=0).last()
    item_id_series = pd.Series(item_ids, index=future_cov.index)
    for col in known_covariates_names:
        future_cov[col] = hist_cov[col].reindex(future_cov.index)
        future_cov[col] = future_cov[col].fillna(item_id_series.map(last_by_item[col]))
    return future_cov
