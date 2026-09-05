"""Tests for scripts/hpa_dataset/preprocessing/generate_tissue_masks.py."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import generate_tissue_masks as G


def _make_tif(path, tissue=True, size=400):
    img = np.full((size, size, 3), 255, np.uint8)          # white background
    if tissue:
        img[100:300, 120:320] = (120, 60, 40)               # dark brown blob
    Image.fromarray(img).save(path)


def test_mask_has_tissue_and_shape(tmp_path):
    p = tmp_path / "a.tif"
    _make_tif(p)
    with Image.open(p) as im:
        mask = G.fast_tissue_mask(im, downsample=0.5, min_component_area=50)
    assert mask.shape == (400, 400)
    assert mask[200, 220] == 255 and mask[10, 10] == 0


def test_root_mode_reports_and_exit_code(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    _make_tif(root / "ok.tif")
    _make_tif(root / "blank.tif", tissue=False)
    (root / "corrupt.tif").write_bytes(b"not a tiff")
    rc = G.main(["--root", str(root), "--num-workers", "1", "--output-format", "npy",
                 "--downsample", "0.5", "--min-component-area", "50"])
    assert rc == 1                                           # one failure
    masks = sorted(p.name for p in (root / "masks").glob("*_mask.npy"))
    assert masks == ["blank_mask.npy", "ok_mask.npy"]
    assert np.load(root / "masks" / "ok_mask.npy").max() == 1
    failed = list(root.glob("generate_tissue_masks_failed_*.csv"))
    no_tissue = list(root.glob("generate_tissue_masks_no_tissue_*.csv"))
    assert len(failed) == 1 and len(no_tissue) == 1
    assert pd.read_csv(failed[0])["image_path"].iloc[0].endswith("corrupt.tif")
    assert pd.read_csv(no_tissue[0])["image_path"].iloc[0].endswith("blank.tif")


def test_csv_mode_and_zero_byte_mask_is_redone(tmp_path):
    root = tmp_path / "hpa"
    sub = root / "batch" / "1"
    sub.mkdir(parents=True)
    _make_tif(sub / "x.tif")
    (sub / "masks").mkdir()
    (sub / "masks" / "x_mask.npy").write_bytes(b"")           # partial write from a killed job
    csv = tmp_path / "ds.csv"
    pd.DataFrame({"local_path": ["batch/1", "batch/1"], "image_id": ["x", "missing"]}).to_csv(csv, index=False)
    rc = G.main(["--csv-file", str(csv), "--img-root", str(root), "--num-workers", "1",
                 "--downsample", "0.5", "--min-component-area", "50"])
    assert rc == 0
    assert (sub / "masks" / "x_mask.npy").stat().st_size > 0


def test_requires_one_input_source():
    with pytest.raises(SystemExit):
        G.main([])
