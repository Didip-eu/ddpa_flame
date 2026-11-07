"""flame.web: texts in, one ZIP of results out."""

import json
import zipfile

from flame import web

FORMULA = "in nomine sancte et individue trinitatis amen notum sit omnibus presentibus et futuris quod nos "
TEXTS = [
    {"id": "a1", "year": "1182", "body": FORMULA + "dedimus ecclesie sancti petri curiam nostram in villa"},
    {"id": "a2", "year": None, "body": FORMULA + "confirmamus monasterio sancte marie decimam de agris"},
    {"id": "a3", "year": "1290", "body": "ich jacob ebser vergich mit dem brief allen den die in sehent oder horent lesen das mir"},
]


def test_run_writes_one_zip_with_pairs_and_record(tmp_path):
    given = {"a": TEXTS, "b": None, "names": {"a1": "Charter one"}, "record": {"kind": "tenor"}}
    options = {"min_text_length": 20, "similarity_threshold": 0.1, "gen_spans_jsonl": True, "gen_heatmap": False}
    paths = json.loads(web.run(json.dumps(given), json.dumps(options), str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        names = set(zipped.namelist())
        pairs = zipped.read("pairs.tsv").decode().splitlines()
        record = json.loads(zipped.read("run.json"))
    assert {"pairs.tsv", "run.json", "similarity_spans.jsonl"} <= names
    assert pairs[1].split("\t")[:4] == ["a1", "Charter one", "a2", "a2"]
    assert record["kind"] == "tenor" and record["texts"] == {"first": 3, "second": 0}
