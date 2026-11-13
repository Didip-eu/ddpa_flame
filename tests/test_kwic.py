"""The KWIC export: one row per occurrence, with offsets that slice the keyword out of the file."""

import csv
import io
import json
import zipfile

from flame import analysis, fuzz, web

FORMULA = "damus et concedimus ac presentibus confirmamus pro remedio anime nostre"
TEXTS = [
    {"id": f"{1300 + n}-{name}.txt", "year": str(1300 + n),
     "body": f"{head} {FORMULA} {tail}"}
    for n, (name, head, tail) in enumerate([
        ("a", "nos karolus rex et omnes fideles curie nostre", "ecclesie sancti petri in villa"),
        ("b", "ego fridericus dux cum consilio baronum meorum", "monasterio sancte marie de agris"),
        ("c", "nos albertus comes de consensu fratrum nostrorum", "capelle sancti georgii in monte"),
    ])
]


def test_kwic_offsets_slice_the_keyword_out_of_the_file(tmp_path):
    options = {"min_text_length": 20, "similarity_threshold": 0.1, "gen_heatmap": False,
               "gen_kwic": True, "kwic_tei": True, "kwic_query": "damus et concedimus ac presentibus confirmamus"}
    paths = json.loads(web.run(json.dumps({"a": TEXTS, "b": None}), json.dumps(options), str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        assert "kwic_tei.xml" in zipped.namelist()
        rows = list(csv.DictReader(io.StringIO(zipped.read("kwic.tsv").decode()), delimiter="\t"))
        assert len(rows) == len(TEXTS)
    for row in rows:
        assert row["Basis"] == "file"
        with open(tmp_path / "run" / "a" / row["File"], encoding="utf-8", newline="") as f:
            assert f.read()[int(row["CharStart"]):int(row["CharEnd"])] == row["Keyword"]
    assert rows and all(row["Keyword"].startswith("damus") for row in rows)


def test_kwic_takes_a_phrase_written_otherwise_with_the_pure_python_fuzz(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "fuzz", fuzz)
    options = {"min_text_length": 20, "similarity_threshold": 0.1, "gen_heatmap": False,
               "gen_kwic": True, "kwic_query": "damus et concedimus ac presentibus confirmavimus"}
    paths = json.loads(web.run(json.dumps({"a": TEXTS, "b": None}), json.dumps(options), str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        rows = list(csv.DictReader(io.StringIO(zipped.read("kwic.tsv").decode()), delimiter="\t"))
    assert len(rows) == len(TEXTS)
    assert all(0.85 <= float(row["QueryScore"]) < 1 for row in rows)
