"""New dataset: ISS + FPFH/SHOT + LightGlue.

Public interface:
    from gluefactory.datasets.new_dataset import (
        NewDatasetISSDescDataset, collate_fn_dynamic_pad,
    )
"""

from .dataset import NewDatasetISSDescDataset, collate_fn_dynamic_pad

__all__ = ["NewDatasetISSDescDataset", "collate_fn_dynamic_pad"]
