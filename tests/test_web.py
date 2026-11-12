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


def test_pairs_of_two_texts_of_one_document_leave_pairs_tsv_unless_asked_for(tmp_path):
    read = FORMULA + "dedimus ecclesie sancti petri curiam nostram in vila"
    texts = [{**TEXTS[0], "id": "d1:tenor", "document": "d1"}, {"id": "d1:tenor:reader", "year": "1182", "body": read, "document": "d1"},
             {**TEXTS[1], "id": "d2:tenor", "document": "d2"}]
    options = {"min_text_length": 20, "similarity_threshold": 0.1, "gen_spans_jsonl": True, "gen_heatmap": False}

    def pairs(self_pairs):
        given = {"a": texts, "b": None, "names": {}, "self_pairs": self_pairs}
        paths = json.loads(web.run(json.dumps(given), json.dumps(options), str(tmp_path / str(self_pairs))))
        with zipfile.ZipFile(paths["archive"]) as zipped:
            rows = [line.split("\t") for line in zipped.read("pairs.tsv").decode().splitlines()[1:]]
            return {(row[0], row[2]) for row in rows}, json.loads(zipped.read("run.json"))["self_pairs"]

    kept, kept_recorded = pairs(True)
    left, left_recorded = pairs(False)
    assert ("d1:tenor", "d1:tenor:reader") in kept and kept_recorded is True
    assert left == kept - {("d1:tenor", "d1:tenor:reader")} and left and left_recorded is False
