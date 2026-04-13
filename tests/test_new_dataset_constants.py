from pathlib import Path

from gluefactory.datasets.new_dataset import constants as C


def test_constants_values():
    assert C.LAT_MM == 0.056
    assert C.TRANS_MM == 0.056
    assert C.VERT_MM == 0.0085
    assert C.RESIZE_FACTOR == 0.5
    assert C.N_SCENES == 641
    assert C.DESC_DIMS == {"fpfh": 33, "shot": 352}
    assert isinstance(C.DEFAULT_DATA_ROOT, Path)
    assert str(C.DEFAULT_DATA_ROOT) == "/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output"
