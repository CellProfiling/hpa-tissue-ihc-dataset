"""Tests for scripts/hpa_dataset/preprocessing/build_annotations.py."""
import pandas as pd

import build_annotations as A


def _dataset():
    return pd.DataFrame([
        dict(image_id="i1", antibody_id="HPA1", ensembl_id="G1", gene_name="A", tissue="Colon", organ="GI", patient_id="p1", set="train"),
        dict(image_id="i2", antibody_id="HPA1", ensembl_id="G1", gene_name="A", tissue="Colon", organ="GI", patient_id="p2", set="train"),
        dict(image_id="i3", antibody_id="HPA2", ensembl_id="G2", gene_name="B", tissue="Lung", organ="Lung", patient_id="p3", set="test"),
    ])


def _cells():
    return pd.DataFrame([
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", cell_type="Glandular cells", staining="high", intensity="strong", quantity=">75%", location="nuclear"),
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", cell_type="Glandular cells", staining="high", intensity="strong", quantity=">75%", location="nuclear"),  # exact duplicate
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", cell_type="Endothelial cells", staining="not detected", intensity="negative", quantity="none", location="none"),
        dict(antibody_id="HPA1", ensembl_id="G9", tissue="Colon", cell_type="Glandular cells", staining="low", intensity="weak", quantity="<25%", location="none"),  # other gene, must not join
        dict(antibody_id="HPA2", ensembl_id="G2", tissue="Lung", cell_type="Macrophages", staining="medium", intensity="moderate", quantity="25%-75%", location="cytoplasmic/membranous, nuclear"),
        dict(antibody_id="HPA2", ensembl_id="G2", tissue="Lung", cell_type="Pneumocytes", staining="not representative", intensity="", quantity="", location=""),
    ])


def test_cell_annotations_join_codes_and_dedupe():
    out = A.build_cell_annotations(_dataset(), _cells())
    assert len(out) == 6                                   # 2 images x 2 cell types + 1 image x 2
    i1 = out[out.image_id == "i1"].set_index("cell_type")
    assert i1.loc["Glandular cells", ["staining_code", "intensity_code", "quantity_code", "location_code"]].tolist() == [3, 3, 3, 2]
    assert i1.loc["Endothelial cells", "staining_code"] == 0
    i3 = out[out.image_id == "i3"].set_index("cell_type")
    assert i3.loc["Macrophages", ["quantity_code", "location_code"]].tolist() == [2, 3]
    assert i3.loc["Pneumocytes", ["staining_code", "intensity_code", "location_code"]].tolist() == [-1, -1, -1]
    assert "G9" not in set(out.ensembl_id)


def test_image_annotations_and_vocabulary():
    ds = _dataset()
    cells = A.build_cell_annotations(ds, _cells())
    patients = pd.DataFrame([
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", patient_id="p1", sex="Female", age="61"),
        dict(antibody_id="HPA2", ensembl_id="G2", tissue="Lung", patient_id="p3", sex="Male", age="40"),
    ])
    tissues = pd.DataFrame([
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", organ="GI", ontology_terms="UBERON:0001155"),
        dict(antibody_id="HPA1", ensembl_id="G1", tissue="Colon", organ="GI", ontology_terms="UBERON:0000059"),
    ])
    img = A.build_image_annotations(ds, cells, patients, tissues).set_index("image_id")
    assert len(img) == 3
    assert img.loc["i1", ["sex", "age", "ontology_terms"]].tolist() == ["Female", "61", "UBERON:0000059;UBERON:0001155"]
    assert img.loc["i2", "sex"] == ""                       # no patient row -> empty, image kept
    assert img.loc["i1", "cell_types"] == "Endothelial cells;Glandular cells"
    assert img.loc["i1", "max_intensity_code"] == 3 and bool(img.loc["i1", "protein_detected"])
    assert img.loc["i3", "protein_detected"] and img.loc["i3", "max_staining_code"] == 2
    voc = A.build_cell_type_vocabulary(cells).set_index("cell_type")
    assert voc.loc["Glandular cells", "n_images"] == 2 and voc.loc["Macrophages", "n_antibodies"] == 1


def test_main_writes_three_files(tmp_path):
    ds = tmp_path / "ds.csv"; _dataset().to_csv(ds, index=False)
    (tmp_path / "meta").mkdir(); _cells().to_csv(tmp_path / "meta" / "cells.csv", index=False)
    assert A.main(["--dataset-csv", str(ds), "--metadata-dir", str(tmp_path / "meta")]) == 0
    assert {p.name for p in tmp_path.glob("ds_*.csv")} == {"ds_cell_annotations.csv", "ds_image_annotations.csv", "ds_cell_types.csv"}
