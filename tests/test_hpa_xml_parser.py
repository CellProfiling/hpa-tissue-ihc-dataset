import json
from pathlib import Path

import pandas as pd

from tests.conftest import make_xml

import hpa_xml_parser as P


def _parse_to_csv(tmp_path, xml_text):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    xml_file = tmp_path / "in.xml"
    xml_file.write_text(xml_text, encoding="utf-8")
    out_dir = tmp_path / "out"
    P.extract_hpa_data(str(xml_file), str(out_dir), "csv", None, str(tmp_path / "log.txt"))
    return pd.read_csv(out_dir / "metadata" / "images.csv", dtype=str, keep_default_na=False)


def test_basic_parse_prefers_tif_and_keeps_patient(tmp_path):
    df = _parse_to_csv(tmp_path, make_xml())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["patient_id"] == "4016"
    assert row["image_type"] == "tif"
    assert row["image_url"].endswith("100007_A_1_8.tif")
    assert row["image_id"] == "100007_A_1_8"
    assert row["download_status"] == "pending"
    assert row["local_path"] == "20250612-xml/1"


def test_patient_and_urls_survive_block_boundary(tmp_path):
    # iterparse feeds 16 KiB blocks. The header before <patientId> is ~750 bytes,
    # so pads in this range walk the tag across the 16384-byte boundary.
    bad = []
    for pad_len in range(15500, 15800):
        df = _parse_to_csv(tmp_path / str(pad_len), make_xml(pad_len))
        if len(df) != 1 or df.loc[0, "patient_id"] != "4016" or df.loc[0, "image_type"] != "tif":
            bad.append((pad_len, df.to_dict("records")))
    assert not bad, f"{len(bad)} pad lengths lost fields; first: {bad[:2]}"


def test_batch_assigner_rolls_over_every_500_parents():
    a = P.BatchFolderAssigner()
    assert a.current_batch_id == "20250612-xml"
    for i in range(499):
        assert a.grandparent_for(f"p{i}") == "20250612-xml"
    # 500th new parent is placed in batch 12, then the counter rolls over
    assert a.grandparent_for("p499") == "20250612-xml"
    assert a.grandparent_for("p500") == "20250613-xml"
    # repeated parent keeps its original batch and does not advance the counter
    assert a.grandparent_for("p0") == "20250612-xml"
    assert a.next_batch_id == 13


def test_resolve_image_folders_web_and_ftp():
    a = P.BatchFolderAssigner()
    url = "http://images.proteinatlas.org/1/100007_A_1_8.jpg"
    assert P.resolve_image_folders(url, "HPA000001", "100007_A_1_8", None, a) == (
        "20250612-xml", "1", "web")
    ftp = {"100007_A_1_8": "idr0043/20250101-ftp/1/100007_A_1_8.tif"}
    assert P.resolve_image_folders(url, "HPA000001", "100007_A_1_8", ftp, a) == (
        "20250101-ftp", "1", "ftp")
    # invalid (too short) ftp path falls back to web batching, on a fresh assigner so
    # this actually exercises registration rather than hitting an already-seen parent
    a2 = P.BatchFolderAssigner()
    ftp_bad = {"100007_A_1_8": "100007_A_1_8.tif"}
    assert P.resolve_image_folders(url, "HPA000001", "100007_A_1_8", ftp_bad, a2) == (
        "20250612-xml", "1", "web")
    assert a2.parent_folders == {"1": "20250612-xml"}
    assert a2.next_batch_id == 12
    # URL without a parent segment falls back to antibody numeric id
    assert P.resolve_image_folders("100007_A_1_8.jpg", "HPA000123", "100007_A_1_8", None, a)[1] == "123"


def test_save_as_csv_dedups_genes_once(tmp_path):
    ab = {
        "id": "HPA000001",
        "gene": {"ensembl_id": "ENSG1", "name": "G1", "synonyms": ["a", "b"]},
        "reliability_score": "approved",
        "tissues": [],
    }
    ab2 = dict(ab, id="HPA000002")
    P.save_as_csv([ab, ab2], str(tmp_path), None, str(tmp_path / "log.txt"))
    genes = pd.read_csv(tmp_path / "metadata" / "genes.csv")
    assert len(genes) == 1
    assert genes.loc[0, "synonyms"] == "a, b"


def _antibody_with_two_tissues():
    patient = {"sex": "Male", "age": "44", "patient_id": "4016",
               "images": [{"type": "tif", "url": "http://images.proteinatlas.org/1/100007_A_1_8.tif"}]}
    cells_a = [{"cell_type": "adipocytes", "staining": "medium", "intensity": "moderate",
                "quantity": ">75%", "location": "cytoplasmic/membranous"},
               {"cell_type": "fibroblasts", "staining": "not detected", "intensity": "negative",
                "quantity": None, "location": None}]
    cells_b = [{"cell_type": "peripheral nerve", "staining": "low", "intensity": "weak",
                "quantity": "<25%", "location": "nuclear"}]
    return {
        "id": "HPA000001",
        "gene": {"ensembl_id": "ENSG1", "name": "G1", "synonyms": []},
        "reliability_score": "approved",
        "tissues": [
            {"tissue": "Adipose tissue", "organ": "Soft tissue", "ontology_terms": "U1",
             "tissue_cells": cells_a, "patients": [patient]},
            {"tissue": "Soft tissue 1", "organ": "Soft tissue", "ontology_terms": "U2",
             "tissue_cells": cells_b, "patients": [patient]},
        ],
    }


def test_image_json_aggregates_all_tissues_and_cells(tmp_path):
    n, errs = P.save_image_metadata([_antibody_with_two_tissues()], str(tmp_path), None,
                                    str(tmp_path / "log.txt"))
    assert errs == 0
    files = list(tmp_path.glob("*/*/metadata/*.json"))
    assert [f.name for f in files] == ["100007_A_1_8.json"]
    meta = json.loads(files[0].read_text())
    assert meta["patient_id"] == "4016"
    assert meta["image_type"] == "tif"
    assert [t["tissue"] for t in meta["tissues"]] == ["Adipose tissue", "Soft tissue 1"]
    assert [c["cell_type"] for c in meta["tissues"][0]["tissue_cells"]] == ["adipocytes", "fibroblasts"]
    assert meta["tissues"][0]["tissue_cells"][0]["quantity"] == ">75%"


def test_same_named_tissues_with_different_ontology_both_survive(tmp_path):
    # Two <data> blocks share a tissue NAME but differ in ontology_terms and cells.
    # The dedup guard must key on (tissue, organ, ontology_terms), not tissue alone,
    # or the second annotation is silently dropped.
    patient = {"sex": "Male", "age": "44", "patient_id": "4016",
               "images": [{"type": "tif", "url": "http://images.proteinatlas.org/1/100007_A_1_8.tif"}]}
    cells_u1 = [{"cell_type": "adipocytes", "staining": "medium", "intensity": "moderate",
                 "quantity": ">75%", "location": "cytoplasmic/membranous"}]
    cells_u2 = [{"cell_type": "fibroblasts", "staining": "low", "intensity": "weak",
                 "quantity": "<25%", "location": "nuclear"}]
    antibody = {
        "id": "HPA000001",
        "gene": {"ensembl_id": "ENSG1", "name": "G1", "synonyms": []},
        "reliability_score": "approved",
        "tissues": [
            {"tissue": "Soft tissue 1", "organ": "Soft tissue", "ontology_terms": "U1",
             "tissue_cells": cells_u1, "patients": [patient]},
            {"tissue": "Soft tissue 1", "organ": "Soft tissue", "ontology_terms": "U2",
             "tissue_cells": cells_u2, "patients": [patient]},
            # A truly identical repeat of the first block must still de-dup to one entry.
            {"tissue": "Soft tissue 1", "organ": "Soft tissue", "ontology_terms": "U1",
             "tissue_cells": cells_u1, "patients": [patient]},
        ],
    }
    n, errs = P.save_image_metadata([antibody], str(tmp_path), None, str(tmp_path / "log.txt"))
    assert errs == 0
    files = list(tmp_path.glob("*/*/metadata/*.json"))
    meta = json.loads(files[0].read_text())
    assert len(meta["tissues"]) == 2
    assert [t["ontology_terms"] for t in meta["tissues"]] == ["U1", "U2"]
    assert meta["tissues"][0]["tissue_cells"][0]["cell_type"] == "adipocytes"
    assert meta["tissues"][1]["tissue_cells"][0]["cell_type"] == "fibroblasts"
