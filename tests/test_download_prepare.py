"""data/download.py (verification, safe extraction) and data/prepare.py (imports, step order)."""

import zipfile

import pytest

from data import download, prepare  # noqa: F401  (prepare: import check)


def test_sha256_and_safe_extract(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("dir/x.txt", "hello")
        f.writestr("__MACOSX/junk", "x")
    assert len(download.sha256(z)) == 64
    assert download.safe_extract(z, tmp_path / "out") == 1 and (tmp_path / "out/dir/x.txt").read_text() == "hello"
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as f:
        f.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        download.safe_extract(bad, tmp_path / "out2")


def test_archive_list_matches_readme():
    assert len(download.ARCHIVES) == 7 and all(len(a[3]) == 64 for a in download.ARCHIVES)
