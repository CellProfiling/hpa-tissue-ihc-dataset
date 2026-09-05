"""Tests for scripts/hpa_dataset/download/download_dataset_images.py (fake HTTP)."""
import pandas as pd
import pytest

import download_dataset_images as D

TIF = b"II*\x00" + b"\x00" * 2000
IDR = "https://ftp.ebi.ac.uk/x/1_A_1_1.tif"
HPA = "https://images.proteinatlas.org/1/1_A_1_1.tif"


class Resp:
    def __init__(self, status=200, body=TIF, ctype="image/tiff"):
        self.status_code, self._body, self.headers = status, body, {"content-type": ctype}

    def iter_content(self, chunk_size=1):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Session:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = responses, [], {}

    def get(self, url, **kw):
        self.calls.append(url)
        return self.responses.get(url, Resp(404, b"", "text/html"))


def _row(**kw):
    r = {"image_id": "1_A_1_1", "local_path": "b/1", "idr_url": IDR, "hpa_url": HPA}
    r.update(kw)
    return r


def test_idr_first_then_hpa_fallback(tmp_path):
    s = Session({IDR: Resp(404, b"", "text/html"), HPA: Resp()})
    res = D.download_row(_row(), str(tmp_path), s, ["idr", "hpa"])
    assert res["status"] == "ok" and res["source"] == "hpa"
    assert s.calls == [IDR, HPA]                        # 404 is not retried
    assert D.is_valid_tif(tmp_path / "b" / "1" / "1_A_1_1.tif")


def test_existing_valid_file_is_skipped_and_bad_body_rejected(tmp_path):
    s = Session({IDR: Resp(200, b"<html>rate limited</html>" * 100, "text/html")})
    res = D.download_row(_row(), str(tmp_path), s, ["idr"], retries=0)
    assert res["status"] == "failed" and "not an image" in res["message"]
    assert not list((tmp_path / "b" / "1").glob("*.part"))
    (tmp_path / "b" / "1" / "1_A_1_1.tif").write_bytes(TIF)
    res = D.download_row(_row(), str(tmp_path), Session({}), ["idr"])
    assert res["status"] == "skipped"


def test_main_filters_sets_and_reports(tmp_path, monkeypatch):
    csv = tmp_path / "ds.csv"
    pd.DataFrame([_row(image_id="a", set="train"), _row(image_id="b", set="test", idr_url="", hpa_url="")]).to_csv(csv, index=False)
    monkeypatch.setattr(D.requests, "Session", lambda: Session({IDR: Resp()}))
    rc = D.main(["--csv", str(csv), "--root", str(tmp_path / "out"), "--set", "train", "--workers", "1"])
    assert rc == 0 and (tmp_path / "out" / "b" / "1" / "a.tif").exists()
    rc = D.main(["--csv", str(csv), "--root", str(tmp_path / "out"), "--workers", "1"])
    assert rc == 1                                       # row b has no url
    assert list((tmp_path / "out").glob("download_failed_*.csv"))


def test_rejects_unknown_source(tmp_path):
    with pytest.raises(SystemExit):
        D.main(["--csv", "x.csv", "--root", str(tmp_path), "--source-order", "ftp"])


def _write_real_tif(path, size=(40, 30)):
    from PIL import Image
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 128, dtype="uint8")).save(path, format="TIFF")


def test_verify_reports_missing_and_corrupt_and_can_delete(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write_real_tif(root / "b" / "1" / "good.tif")
    (root / "b" / "1").mkdir(parents=True, exist_ok=True)
    (root / "b" / "1" / "bad.tif").write_bytes(TIF)            # magic ok, undecodable
    csv = tmp_path / "ds.csv"
    pd.DataFrame([_row(image_id="good"), _row(image_id="bad"), _row(image_id="absent")]).to_csv(csv, index=False)

    rc = D.main(["--csv", str(csv), "--root", str(root), "--verify", "--workers", "1"])
    assert rc == 1 and (root / "b" / "1" / "bad.tif").exists()
    report = pd.read_csv(next(root.glob("verify_failed_*.csv")))
    assert dict(zip(report.image_id, report.status)) == {"bad": "corrupt", "absent": "missing"}

    rc = D.main(["--csv", str(csv), "--root", str(root), "--verify", "--delete-corrupt", "--workers", "1"])
    assert rc == 1 and not (root / "b" / "1" / "bad.tif").exists()
    assert D.verify_row(_row(image_id="good"), str(root))["status"] == "ok"
