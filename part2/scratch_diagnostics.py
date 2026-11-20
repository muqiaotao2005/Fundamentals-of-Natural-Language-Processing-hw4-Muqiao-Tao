"""Diagnostics shared by the from-scratch short-run stages."""
import hashlib
import json
import math
import pickle
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from load_data import INPUT_PREFIX, T5Dataset, load_lines, normal_collate_fn
from utils import compute_record_F1, compute_record_exact_match, compute_sql_exact_match


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


class PlainRecords(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Expected only plain database records')


def read_reference(data_dir):
    with (Path(data_dir).parent / 'records/ground_truth_dev.pkl').open('rb') as f:
        records, errors = PlainRecords(f).load()
    if any(errors):
        raise ValueError('Reference records contain execution errors')
    return records


def load_split(data_dir, split):
    # Deliberately do not load the held-out test split in a diagnostic run.
    if split not in ('train', 'dev'):
        raise ValueError('Short runs only use train/dev')
    data_dir = Path(data_dir)
    questions = load_lines(data_dir / f'{split}.nl')
    sql = load_lines(data_dir / f'{split}.sql')
    if not questions or len(questions) != len(sql) or not all(questions) or not all(sql):
        raise ValueError(f'{split}: empty or misaligned input/target data')
    return T5Dataset(str(data_dir), split), questions, sql


def loader(dataset, indices, batch_size, shuffle=False, generator=None):
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0,
                      collate_fn=normal_collate_fn)


def fingerprint(data_dir, tokenizer):
    data_dir = Path(data_dir)
    paths = [data_dir / name for name in
             ('train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'flight_database.db')]
    paths.append(data_dir.parent / 'records/ground_truth_dev.pkl')
    paths += [Path(__file__).with_name(name) for name in
              ('load_data.py', 'utils.py', 'scratch_diagnostics.py', 'train_t5_scratch.py')]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes['tokenizer'] = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def validate_tokenizer(tokenizer, questions, queries, max_new_tokens):
    encoded = tokenizer(queries, truncation=False)['input_ids']
    decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    mismatches = [i+1 for i, (a, b) in enumerate(zip(queries, decoded)) if a != b]
    unknown = [i+1 for i, row in enumerate(encoded) if tokenizer.unk_token_id in row]
    if mismatches or unknown:
        raise ValueError(f'Lossy SQL tokenization: roundtrip lines={mismatches[:10]}, '
                         f'UNK lines={unknown[:10]}')
    if not all(row[-1] == tokenizer.eos_token_id for row in encoded):
        raise ValueError('Missing EOS in SQL targets')
    maximum = max(map(len, encoded))
    if maximum > max_new_tokens:
        raise ValueError(f'Gold SQL needs {maximum} new tokens including EOS; '
                         f'configured limit is {max_new_tokens}')
    source = tokenizer([INPUT_PREFIX + q for q in questions], truncation=False)['input_ids']
    return dict(examples=len(queries), roundtrip_mismatches=0, sql_unk_examples=0,
                input_unk_examples=sum(tokenizer.unk_token_id in row for row in source),
                max_sql_tokens_with_eos=maximum,
                max_input_tokens_with_eos=max(map(len, source)))


def masked_loss(model, batch, device, reduction='mean'):
    source, mask, decoder, targets, _ = [t.to(device) for t in batch]
    logits = model(input_ids=source, attention_mask=mask,
                   decoder_input_ids=decoder, use_cache=False).logits
    valid = targets.ne(0)
    loss = F.cross_entropy(logits[valid], targets[valid], reduction=reduction)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite cross-entropy loss')
    correct = (logits.detach().argmax(-1)[valid] == targets[valid]).sum().item()
    return loss, valid.sum().item(), correct


@torch.no_grad()
def loss_stats(model, batches, device):
    model.eval()
    total, count, correct = 0.0, 0, 0
    for batch in batches:
        loss, n, hit = masked_loss(model, batch, device, reduction='sum')
        total += loss.item()
        count += n
        correct += hit
    return dict(loss=total/count, token_accuracy=correct/count, target_tokens=count)


def batch_check(model, batch, tokenizer, device):
    source, mask, decoder, targets, initial = [t.to(device) for t in batch]
    assert torch.equal(mask.bool(), source.ne(tokenizer.pad_token_id))
    assert torch.all(initial == model.config.decoder_start_token_id)
    assert model.config.decoder_start_token_id == tokenizer.pad_token_id == 0
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
    assert model.get_output_embeddings().weight.shape[0] == len(tokenizer)
    assert all(p.requires_grad for p in model.parameters())
    for row in range(len(targets)):
        n = int(targets[row].ne(0).sum())
        assert targets[row, n-1].item() == tokenizer.eos_token_id
        assert torch.equal(decoder[row, 1:n], targets[row, :n-1])
    model.eval()  # Disable dropout when comparing two equivalent loss paths.
    with torch.no_grad():
        manual, _, _ = masked_loss(model, batch, device)
        native = model(input_ids=source, attention_mask=mask,
                       labels=targets.masked_fill(targets.eq(0), -100)).loss
        assert torch.allclose(manual, native, atol=1e-5), (manual, native)
    parameter = model.encoder.block[0].layer[0].SelfAttention.q.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    model.train()
    loss, _, _ = masked_loss(model, batch, device)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    assert math.isfinite(float(norm)) and norm > 0
    optimizer.step()
    assert not torch.equal(before, parameter)
    return dict(shift_and_masks=True, manual_matches_native_loss=True,
                initial_loss=float(manual), finite_gradient_norm=float(norm),
                parameters_updated=True)


@torch.no_grad()
def generate(model, batches, tokenizer, device, max_new_tokens):
    model.eval()
    queries, diagnostics = [], []
    for source, mask, _, _, initial in tqdm(batches, desc='Generate', leave=False):
        output = model.generate(input_ids=source.to(device), attention_mask=mask.to(device),
                                decoder_input_ids=initial.to(device), do_sample=False,
                                num_beams=1, max_new_tokens=max_new_tokens)
        queries.extend(q.replace('\n', ' ').replace('\r', ' ').strip() for q in
                       tokenizer.batch_decode(output, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False))
        for sequence in output.tolist():
            tokens = sequence[1:]  # Exclude decoder start, retain EOS.
            eos = tokenizer.eos_token_id in tokens
            length = tokens.index(tokenizer.eos_token_id)+1 if eos else len(tokens)
            diagnostics.append(dict(has_eos=eos, generated_tokens=length,
                                    hit_length_limit=not eos and length >= max_new_tokens))
    n = len(queries)
    stats = dict(eos_rate=sum(d['has_eos'] for d in diagnostics)/n,
                 length_limit_rate=sum(d['hit_length_limit'] for d in diagnostics)/n,
                 empty_rate=sum(not q for q in queries)/n,
                 distinct_query_fraction=len(set(queries))/n,
                 mean_generated_tokens=sum(d['generated_tokens'] for d in diagnostics)/n)
    return queries, diagnostics, stats


def execute_query(db_path, query, timeout_seconds, stop_event=None):
    if not query.lstrip().upper().startswith('SELECT '):
        return [], 'RejectedQuery: expected a SELECT query', 0.0
    start = time.monotonic()
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.execute('PRAGMA query_only = ON')
    conn.set_progress_handler(lambda: int(time.monotonic()-start > timeout_seconds or
                               (stop_event is not None and stop_event.is_set())), 1000)
    try:
        records, error = conn.execute(query).fetchall(), ''
    except sqlite3.Error as exc:
        records, error = [], f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()
    return records, error, time.monotonic()-start


def execute_queries(db_path, queries, timeout_seconds=30, workers=2):
    # Deduplicate execution only. Preserve every original query and its position.
    unique = list(dict.fromkeys(queries))
    completed = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    stop = threading.Event()
    futures = {pool.submit(execute_query, db_path, q, timeout_seconds, stop): q for q in unique}
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc='SQL', leave=False):
            completed[futures[future]] = future.result()
    except BaseException:
        stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return tuple([[completed[q][column] for q in queries] for column in range(3)])


def error_kind(error):
    if not error:
        return 'none'
    if 'interrupted' in error:
        return 'timeout'
    if 'no such' in error or 'ambiguous' in error:
        return 'name_resolution'
    if 'syntax' in error or 'incomplete input' in error:
        return 'syntax'
    return 'other'


def record_scores(gold_sql, queries, gold_records, records, errors):
    if not len(gold_sql) == len(queries) == len(gold_records) == len(records) == len(errors):
        raise ValueError('Misaligned evaluation examples')
    return dict(sql_em=compute_sql_exact_match(gold_sql, queries),
                record_em=compute_record_exact_match(gold_records, records),
                record_f1=float(compute_record_F1(gold_records, records)),
                execution_error_rate=sum(bool(e) for e in errors)/len(errors),
                execution_error_counts=dict(Counter(error_kind(e) for e in errors)))


def save_predictions(prefix, indices, questions, gold_sql, queries, diagnostics,
                     records=None, errors=None):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.sql').write_text(''.join(q+'\n' for q in queries))
    if records is not None:
        with prefix.with_suffix('.pkl').open('wb') as f:
            pickle.dump((records, errors), f)
    with prefix.with_suffix('.jsonl').open('w') as f:
        for position, i in enumerate(indices):
            row = dict(source_line=i+1, question=questions[i], gold_sql=gold_sql[i],
                       predicted_sql=queries[position], **diagnostics[position])
            if records is not None:
                row.update(record_count=len(set(records[position])),
                           execution_error=errors[position])
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
