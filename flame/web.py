"""FLAME in a browser worker: a run on texts given as JSON, its results as one ZIP."""

import json
import os
import re
import time
import zipfile

from .analysis import parse_args, run as run_analysis

LEADING_YEAR = re.compile(r'^(?:-?\d{4}|undated)_')


def _write_corpus(directory: str, texts: list[dict]) -> None:
    """One file per text, named so that FLAME reads its year from the name."""
    os.makedirs(directory)
    for text in texts:
        name = f"{text.get('year') or 'undated'}_{text['id']}.txt"
        with open(os.path.join(directory, name), 'w', encoding='utf-8', newline='') as f:
            f.write(text['body'])


def _pairs_tsv(spans_path: str, names: dict, documents: dict, self_pairs: bool) -> str:
    """The pairs above the threshold, highest score first, with the documents' ids and names; without the pairs of two
    texts of one document unless self_pairs."""
    rows = ['id_1\tname_1\tid_2\tname_2\tscore\tcluster']
    if not os.path.exists(spans_path):
        return '\n'.join(rows) + '\n'
    with open(spans_path, encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    for pair in sorted(pairs, key=lambda pair: -pair['score']):
        first = LEADING_YEAR.sub('', pair['file_1'].rsplit('.', 1)[0])
        second = LEADING_YEAR.sub('', pair['file_2'].rsplit('.', 1)[0])
        if not self_pairs and documents.get(first, first) == documents.get(second, second):
            continue
        cluster = pair.get('cluster_id')
        rows.append('\t'.join([first, names.get(first, first), second, names.get(second, second),
                               str(pair['score']), '' if cluster is None else str(cluster)]))
    return '\n'.join(rows) + '\n'


def _memory_bytes() -> int | None:
    """The size of the Pyodide runtime's memory, where there is one."""
    try:
        import pyodide_js
    except ImportError:
        return None
    return pyodide_js._module.HEAPU8.length


def run(input_json: str, options_json: str, directory: str) -> str:
    """Runs FLAME on the texts of input_json in a new directory and returns, as JSON, the paths of
    the results ZIP and of similarity_spans.jsonl.

    input_json holds `a` and `b` (null or a list): texts as {id, year, body}, each with the id of
    its `document` where several texts come from one; `names`, document names by id; `self_pairs`,
    false to leave the pairs of two texts of one document out of pairs.tsv; and `record`, fields for
    run.json. options_json holds FLAME parameters by name."""
    given = json.loads(input_json)
    options = json.loads(options_json)
    self_pairs = given.get('self_pairs', True)
    documents = {text['id']: text.get('document', text['id']) for text in given['a'] + (given.get('b') or [])}
    first, second, out = (os.path.join(directory, part) for part in ('a', 'b', 'out'))
    _write_corpus(first, given['a'])
    argv = ['-input_path', first]
    if given.get('b') is not None:
        _write_corpus(second, given['b'])
        argv += ['-input_path2', second]
    for name, value in options.items():
        argv += [f'-{name}', str(value).lower() if isinstance(value, bool) else str(value)]
    os.makedirs(out)

    started = time.perf_counter()
    previous = os.getcwd()
    os.chdir(out)
    try:
        run_analysis(parse_args(argv))
    finally:
        os.chdir(previous)
    record = {**given.get('record', {}), 'texts': {'first': len(given['a']), 'second': len(given.get('b') or [])},
              'options': options, 'self_pairs': self_pairs, 'seconds': round(time.perf_counter() - started, 1),
              'memory_bytes': _memory_bytes()}

    spans = os.path.join(out, 'similarity_spans.jsonl')
    archive = os.path.join(directory, 'results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(os.listdir(out)):
            zipped.write(os.path.join(out, name), name)
        zipped.writestr('pairs.tsv', _pairs_tsv(spans, given.get('names', {}), documents, self_pairs))
        zipped.writestr('run.json', json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return json.dumps({'archive': archive, 'spans': spans if os.path.exists(spans) else None})
