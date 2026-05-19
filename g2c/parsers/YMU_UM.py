"""Backward-compatible wrapper around the unified Tobii parser.

New code should prefer:

    from g2c.parsers import load
    eye_events, samples = load("YMU_UM", sample_size=100)

This file is preserved so that legacy notebooks continue to work
unchanged.
"""
from __future__ import annotations

from .datasets_config import DATASETS
from .load import load as _load


def YMU_UM(sample_size: int = None):
    """Parse the YMU_UM dataset.

    Parameters
    ----------
    sample_size : int, optional
        Number of participant TSV files to read. If None, the default
        from `datasets_config.DATASETS["YMU_UM"]["default_sample_size"]`
        is used (84, preserving the legacy default).

    Returns
    -------
    (eye_events, samples) : tuple of pd.DataFrame
    """
    if sample_size is None:
        sample_size = DATASETS["YMU_UM"]["default_sample_size"]
    return _load("YMU_UM", sample_size=sample_size)
