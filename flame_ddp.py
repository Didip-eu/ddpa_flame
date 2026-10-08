"""Fetches charters from the Monasterium NG API into a folder of texts that FLAME reads."""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

BATCH_LIMIT = 500
RETRIES = 3
TIMEOUT = 30.0
RETRY_STATUSES = {429, 503}
MANIFEST = "manifest.tsv"
FETCH_RECORD = "fetch.json"
MANIFEST_COLUMNS = ("file", "document_id", "atom_id", "year", "text_kind", "text_hash", "legacy_url")


@dataclass
class Entry:
    """One text of the corpus with the facts that tie it back to its document."""
    file: str
    document_id: str
    atom_id: str
    year: str
    text_kind: str
    text_hash: str
    legacy_url: str
    text: str


@dataclass
class Fetch:
    """What one fetch asked for, found and wrote."""
    api: str
    api_version: str
    text_kind: str
    values: list[str]
    unresolved: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    without_text: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)


class Api:
    """The routes of the Monasterium NG API this module reads, over a client whose base URL ends in /api/v1/."""

    def __init__(self, client: httpx.Client):
        self.client = client

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """One request, retried on 429 and 503 after the Retry-After the server sends."""
        response = self.client.request(method, path, **kwargs)
        for attempt in range(1, RETRIES):
            if response.status_code not in RETRY_STATUSES:
                break
            time.sleep(float(response.headers.get("Retry-After", attempt)))
            response = self.client.request(method, path, **kwargs)
        return response

    def version(self) -> str:
        """The version the API reports on its health route."""
        response = self._request("GET", "health")
        response.raise_for_status()
        return response.json()["version"]

    def resolve(self, values: list[str]) -> tuple[list[str], list[str]]:
        """Document ids for document ids, atom ids or Monasterium URLs, and the values that match none."""
        ids, unresolved = [], []
        for start in range(0, len(values), BATCH_LIMIT):
            response = self._request("POST", "documents/resolve", json={"values": values[start:start + BATCH_LIMIT]})
            response.raise_for_status()
            for resolution in response.json()["data"]:
                if resolution["id"]:
                    ids.append(resolution["id"])
                else:
                    unresolved.append(resolution["value"])
        return list(dict.fromkeys(ids)), unresolved

    def document(self, document_id: str) -> dict | None:
        """One document record, or None when the API has no document under that id."""
        response = self._request("GET", f"documents/{document_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def year_of(document: dict) -> str | None:
    """The year the document's date starts in, four digits, or None when it is undated."""
    start = (document["dates"].get("from") or "")[:4]
    return start if len(start) == 4 and start.isdigit() else None


def file_name(document_id: str, year: str | None) -> str:
    """The corpus file name: the year or "undated" first, so FLAME reads the year from it."""
    return f"{year or 'undated'}_{document_id}.txt"


def entry_of(document: dict, text_kind: str) -> Entry | None:
    """The corpus entry of a document's text of one kind, or None when it has no such text."""
    text = next((text for text in document["texts"] if text["kind"] == text_kind and text["body"].strip()), None)
    if text is None:
        return None
    year = year_of(document)
    return Entry(file=file_name(document["id"], year), document_id=document["id"], atom_id=document["atom_id"],
                 year=year or "", text_kind=text_kind, text_hash=text["hash"], legacy_url=document["legacy_url"] or "",
                 text=text["body"])


def fetch(api: Api, values: list[str], text_kind: str, workers: int) -> Fetch:
    """Resolves the values to documents and reads each document's text of one kind."""
    result = Fetch(api=str(api.client.base_url), api_version=api.version(), text_kind=text_kind, values=values)
    ids, result.unresolved = api.resolve(values)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        documents = list(pool.map(api.document, ids))
    for document_id, document in zip(ids, documents):
        if document is None:
            result.not_found.append(document_id)
            continue
        entry = entry_of(document, text_kind)
        if entry is None:
            result.without_text.append(document_id)
            continue
        result.entries.append(entry)
    return result


def write_corpus(out: Path, result: Fetch) -> None:
    """Writes one text file per entry, the manifest, and the record of the fetch into a new folder."""
    out.mkdir(parents=True, exist_ok=True)
    for entry in result.entries:
        (out / entry.file).write_text(entry.text, encoding="utf-8")
    with open(out / MANIFEST, "w", encoding="utf-8", newline="") as manifest:
        writer = csv.writer(manifest, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows([getattr(entry, column) for column in MANIFEST_COLUMNS] for entry in result.entries)
    record = {"api": result.api, "api_version": result.api_version, "text_kind": result.text_kind,
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "values": result.values,
              "written": len(result.entries), "unresolved": result.unresolved, "not_found": result.not_found,
              "without_text": result.without_text}
    (out / FETCH_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def report(result: Fetch) -> str:
    """A few lines on what the fetch found, for the console."""
    lines = [f"Texts of kind '{result.text_kind}' found: {len(result.entries)} for {len(result.values)} values asked for."]
    for label, values in (("Matched no document", result.unresolved), ("Not found", result.not_found),
                          (f"No text of kind '{result.text_kind}'", result.without_text)):
        if values:
            lines.append(f"{label} ({len(values)}): {', '.join(values)}")
    return "\n".join(lines)


def read_values(ids: list[str], ids_file: str) -> list[str]:
    """The values from the command line and the file, split at whitespace, in order, without repeats."""
    lines = Path(ids_file).read_text(encoding="utf-8").splitlines() if ids_file else []
    lines = [line for line in lines if not line.lstrip().startswith("#")]
    values = [value for chunk in ids + lines for value in chunk.split()]
    return list(dict.fromkeys(values))


def run(base_url: str, values: list[str], out: Path, text_kind: str = "tenor", workers: int = 4,
        dry_run: bool = False) -> Fetch:
    """Fetches the documents and, unless dry_run, writes the corpus folder; refuses a folder that is not empty."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a fetch writes into a new folder")
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=TIMEOUT) as client:
        result = fetch(Api(client), values, text_kind, workers)
    print(report(result))
    if dry_run:
        print(f"Dry run: nothing written to {out}.")
    else:
        write_corpus(out, result)
        print(f"Written to {out}: the texts, {MANIFEST} and {FETCH_RECORD}.")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the command line, with one -name option each, as flame.py does."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("-api", required=True, help="the API's base URL, such as https://host/api/v1")
    parser.add_argument("-ids", nargs="*", default=[],
                        help="document ids, atom ids or Monasterium URLs, separated by spaces")
    parser.add_argument("-ids_file", default="", help="a file of such values, one or more per line, # for comments")
    parser.add_argument("-out", required=True, help="the new folder the texts are written to")
    parser.add_argument("-text", default="tenor", choices=("tenor", "abstract"), help="the kind of text (default: tenor)")
    parser.add_argument("-workers", type=int, default=4, help="parallel document requests (default: 4)")
    parser.add_argument("-dry_run", action="store_true", help="fetch and report, write nothing")
    return parser.parse_args(argv)


def main() -> int:
    """Runs a fetch from the command line; the exit code is 0 when it found at least one text."""
    args = parse_args()
    values = read_values(args.ids, args.ids_file)
    if not values:
        print("Give the documents with -ids or -ids_file.", file=sys.stderr)
        return 2
    try:
        result = run(args.api, values, Path(args.out), args.text, args.workers, args.dry_run)
    except (httpx.HTTPError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0 if result.entries else 1


if __name__ == "__main__":
    sys.exit(main())
