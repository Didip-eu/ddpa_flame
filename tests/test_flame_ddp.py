import csv
import json

import httpx
import pytest

import flame_ddp
from flame import SimilarityVisualizer

DATED = "3264c4b39042b535c9d2ba2198aa6b70"
UNDATED = "43616f9f54c66a1663fd66e58ef7baf6"
NO_TENOR = "ea2b6277b87e71b2c95786f7bd8807cc"
GONE = "00000000000000000000000000000000"
URL = "https://www.monasterium.net/mom/AT-X/Y/1/charter"


def document(document_id: str, date_from: str | None, texts: list[dict]) -> dict:
    return {"id": document_id, "atom_id": f"tag:www.monasterium.net,2011:/charter/X/{document_id}",
            "legacy_url": None, "dates": {"kind": "stated", "from": date_from}, "texts": texts}


def tenor(body: str) -> dict:
    return {"id": "t", "kind": "tenor", "language": None, "body": body, "hash": "h-" + body[:4]}


DOCUMENTS = {
    DATED: document(DATED, "1182-02-24", [tenor("In nomine sancte et individue trinitatis")]),
    UNDATED: document(UNDATED, None, [tenor("Nos Rudolfus dei gratia")]),
    NO_TENOR: document(NO_TENOR, "1300", [{"id": "a", "kind": "abstract", "language": "de", "body": "Regest",
                                           "hash": "h"}]),
}


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path.removeprefix("/api/v1/")
    if path == "health":
        return httpx.Response(200, json={"status": "ok", "version": "0.6.0"})
    if path == "documents/resolve":
        values = json.loads(request.content)["values"]
        resolved = {URL: DATED, DATED: DATED, UNDATED: UNDATED, NO_TENOR: NO_TENOR, GONE: GONE}
        return httpx.Response(200, json={"data": [{"value": value, "id": resolved.get(value)} for value in values]})
    document_id = path.removeprefix("documents/")
    if document_id in DOCUMENTS:
        return httpx.Response(200, json=DOCUMENTS[document_id])
    return httpx.Response(404, json={"error": {"code": "not_found"}})


@pytest.fixture
def api():
    with httpx.Client(base_url="http://test/api/v1/", transport=httpx.MockTransport(handler)) as client:
        yield flame_ddp.Api(client)


def test_fetch_sorts_documents_into_texts_and_gaps(api):
    result = flame_ddp.fetch(api, [URL, DATED, UNDATED, NO_TENOR, GONE, "nothing"], "tenor", workers=2)
    assert [entry.file for entry in result.entries] == [f"1182_{DATED}.txt", f"undated_{UNDATED}.txt"]
    assert result.unresolved == ["nothing"]
    assert result.not_found == [GONE]
    assert result.without_text == [NO_TENOR]
    assert result.api_version == "0.6.0"


def test_written_corpus_holds_texts_manifest_and_record(api, tmp_path):
    result = flame_ddp.fetch(api, [DATED, UNDATED], "tenor", workers=1)
    flame_ddp.write_corpus(tmp_path, result)
    assert (tmp_path / f"1182_{DATED}.txt").read_text() == "In nomine sancte et individue trinitatis"
    with open(tmp_path / flame_ddp.MANIFEST) as manifest:
        rows = list(csv.DictReader(manifest, delimiter="\t"))
    assert [(row["document_id"], row["year"]) for row in rows] == [(DATED, "1182"), (UNDATED, "")]
    assert json.loads((tmp_path / flame_ddp.FETCH_RECORD).read_text())["written"] == 2


def test_flame_reads_the_year_of_the_file_name_not_of_the_id():
    year = SimilarityVisualizer._extract_year_from_filename
    assert DATED.find("2198") > 0
    assert year(flame_ddp.file_name(DATED, "1182")) == 1182
    assert year(flame_ddp.file_name(DATED, None)) == 9999


def test_run_refuses_a_folder_that_is_not_empty(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        flame_ddp.run("http://test/api/v1", [DATED], tmp_path)


def test_read_values_splits_and_keeps_order(tmp_path):
    ids_file = tmp_path / "ids.txt"
    atom_id = DOCUMENTS[DATED]["atom_id"]
    ids_file.write_text(f"# a list\n{UNDATED} {DATED}\n{atom_id}\n")
    assert flame_ddp.read_values([DATED, URL], str(ids_file)) == [DATED, URL, UNDATED, atom_id]
