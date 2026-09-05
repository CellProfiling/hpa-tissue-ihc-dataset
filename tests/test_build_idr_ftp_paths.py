"""Tests for scripts/hpa_dataset/preprocessing/build_idr_ftp_paths.py."""
import pandas as pd

import build_idr_ftp_paths as B

ROOT = "/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/"


def _run_csv(path, rows, extra_cols=()):
    cols = B.OUTPUT_COLUMNS + list(extra_cols)
    df = pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)
    df.to_csv(path, index=False)


def _row(image, pathology_acc, extra=()):
    base = [""] * len(B.OUTPUT_COLUMNS)
    base[B.OUTPUT_COLUMNS.index("Image Name")] = image
    base[B.OUTPUT_COLUMNS.index("Characteristics [Pathology] Accession")] = pathology_acc
    base[B.OUTPUT_COLUMNS.index("Characteristics [Pathology]")] = "Normal tissue, NOS;Adenocarcinoma, NOS"
    return base + list(extra)


def test_listing_last_batch_wins_and_ignores_dirs(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("\n".join([
        ROOT, ROOT + "20201116-s3/", ROOT + "20201116-s3/37916/1_A_1_1.tif",
        ROOT + "20210108-ftp/37916/1_A_1_1.tif", ROOT + "20210108-ftp/37916/1_A_1_2.tif",
    ]) + "\n")
    m = B.load_ftp_listing(str(listing))
    assert m == {"1_A_1_1.tif": "20210108-ftp/37916/1_A_1_1.tif",
                 "1_A_1_2.tif": "20210108-ftp/37916/1_A_1_2.tif"}


def test_build_table_filters_normal_and_joins(tmp_path):
    idr = tmp_path / "IDR"
    idr.mkdir()
    _run_csv(idr / "idr0043-experimentA-annotation-run01.csv",
             [_row("1_A_1_1.tif", "M-00100;M-81403", ["v1"]), _row("2_B_1_1.tif", "M-81403", ["v1"])],
             extra_cols=["Ensembl version"])
    _run_csv(idr / "idr0043-experimentA-annotation-run02.csv",
             [_row("3_C_1_1.tif", "M-00100")])
    listing = tmp_path / "list.txt"
    listing.write_text(ROOT + "20180825-ftp/1/1_A_1_1.tif\n")
    table = B.build_table(str(idr), str(listing))
    assert list(table.columns) == B.OUTPUT_COLUMNS + ["source_file", "ftp_path"]
    assert table["Image Name"].tolist() == ["1_A_1_1.tif", "3_C_1_1.tif"]
    assert table["source_file"].tolist() == ["idr0043-experimentA-annotation-run01.csv",
                                             "idr0043-experimentA-annotation-run02.csv"]
    assert table["ftp_path"].tolist() == ["20180825-ftp/1/1_A_1_1.tif", ""]
