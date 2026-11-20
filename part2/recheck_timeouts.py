"""Re-execute timed-out predictions read-only, preserving original artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import pickle
import sqlite3
import time


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Records must contain plain data, not Python objects')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sql', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=Path(__file__).parent/'data/flight_database.db')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.workers < 1:
        parser.error('timeout must be positive and finite; workers must be positive')
    report_path = args.output.with_suffix(args.output.suffix + '.audit.json')
    if args.output.exists() or report_path.exists():
        parser.error('Choose a new output path; existing outputs are protected')
    sql_bytes = args.sql.read_bytes()
    queries = sql_bytes.decode('utf-8').splitlines()
    with args.records.open('rb') as stream:
        value = PlainUnpickler(stream).load()
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError('Expected a (records, errors) tuple')
    records, errors = value
    if not isinstance(records, list) or not isinstance(errors, list):
        raise ValueError('Records and errors must be lists')
    if not len(queries) == len(records) == len(errors) or not all(isinstance(e, str) for e in errors):
        raise ValueError('SQL, records, and errors must align')
    indices = [i for i, error in enumerate(errors)
               if 'interrupted' in error.lower() or 'timeout' in error.lower()
               or 'timed out' in error.lower()]

    def execute(index):
        started = time.monotonic()
        uri = args.database.resolve().as_uri() + '?mode=ro'
        connection = sqlite3.connect(uri, uri=True)
        connection.set_progress_handler(lambda: int(time.monotonic() - started > args.timeout), 1000)
        try:
            rows, error = connection.execute(queries[index]).fetchall(), ''
        except sqlite3.Error as exc:
            rows, error = [], f'{type(exc).__name__}: {exc}'
        finally:
            connection.close()
        return index, rows, error, time.monotonic() - started

    audit = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, rows, error, seconds in pool.map(execute, indices):
            audit.append(dict(line=index+1, previous_error=errors[index], error=error,
                              record_count=len(rows), seconds=seconds))
            records[index], errors[index] = rows, error
    if args.sql.read_bytes() != sql_bytes:
        raise ValueError('SQL changed during execution; refusing to save')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        pickle.dump((records, errors), stream)
    report = dict(sql_sha256=hashlib.sha256(sql_bytes).hexdigest(),
                  rechecked=len(indices), remaining_errors=sum(bool(e) for e in errors),
                  timeout_seconds=args.timeout, workers=args.workers, queries=audit)
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
