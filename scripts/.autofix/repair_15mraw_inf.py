# description: Trace 15mraw infinities, replace them with NaN, then request training
# content: Preserve raw evidence; rebuild invalid dumps and audit every infinite cell before saving.
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Inspect dumps without rebuilding, writing reports or requesting training

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.proj import DB, Logger
from src.proj.util.script import ScriptTool

# Keep the original task's CLI/declaration unchanged; share its backup, save
# verification and maintenance-safe training handoff.
_spec = importlib.util.spec_from_file_location(
    'repair_15mraw_common', Path(__file__).with_name('repair_15mraw_datablocks.py'))
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)
REQUEST_ID = '20261018_repair_15mraw_inf_train_03'


def number(value):
    """Strict JSON: preserve NaN/Inf as strings, not nonstandard JSON numbers."""
    value = float(value)
    return value if np.isfinite(value) else str(value)


def explain_inf(row, feature, rebuilt_value, dtype):
    """Distinguish proven storage/cast origin from a reproducible VWAP mechanism.

    Stored aggregates cannot prove what happened in the vendor's minute feed.
    Report that boundary instead of labelling every VWAP Inf as divide-by-zero.
    """
    raw = float(row[feature])
    result = {'source_value': number(raw), 'rebuilt_value': number(rebuilt_value),
              'source_dtype': dtype, 'origin': 'unexplained_preprocessing_inf',
              'mechanism': 'raw value does not reproduce the rebuilt infinity'}
    if np.isfinite(raw):
        with np.errstate(over='ignore', invalid='ignore'):
            cast = np.float32(raw)
        if np.isinf(cast) and np.signbit(cast) == np.signbit(rebuilt_value):
            result.update(origin='float32_cast_overflow',
                          mechanism='finite raw value overflows float32 during preprocessing')
    elif np.isinf(raw) and np.signbit(raw) == np.signbit(rebuilt_value):
        result.update(origin='stored_raw_inf', mechanism='upstream_origin_unresolved')
        if feature == 'vwap' and 'amount' in row and 'volume' in row:
            amount, volume = float(row['amount']), float(row['volume'])
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                quotient = np.divide(amount, volume)
                cast = np.float32(quotient)
            result['amount_div_volume_float64'] = number(quotient)
            if np.isinf(quotient) and np.signbit(quotient) == np.signbit(raw):
                if np.isfinite(amount) and amount != 0 and volume == 0:
                    result['mechanism'] = 'nonzero_amount_div_zero_volume_reproduces_inf'
                elif not np.isfinite(amount) or not np.isfinite(volume):
                    result['mechanism'] = 'nonfinite_operand_reproduces_inf'
                else:
                    result['mechanism'] = 'float64_division_overflow_reproduces_inf'
            elif np.isfinite(quotient) and np.isinf(cast) and np.signbit(cast) == np.signbit(raw):
                result['mechanism'] = 'float32_division_or_cast_overflow_reproduces_inf'
    result['raw_context'] = {key: number(row[key]) for key in
                             ('open', 'close', 'high', 'low', 'vwap', 'amount', 'volume') if key in row}
    return result


def load_source_day(date):
    # Match the production loader's primary/alternative selection and Polars
    # path. Never overwrite the historical source while repairing derived data.
    path = DB.DBPath('trade_ts', '15min').path(date, use_alt=True)
    frame = DB.loads_pl('trade_ts', '15min', dates=[date], use_alt=True, vb_level='never')
    return frame.to_pandas(), str(path)


def audit_and_sanitize(block, proc, item, folder, *, vwap_only=False, sanitize=True, evidence_name=None):
    """Trace every Inf to its raw row, stream evidence, then replace only Inf.

    Scan one day at a time (bounded memory). Unknown finite-to-Inf transitions
    fail before saving; a stored raw Inf is attributable to its exact source,
    even when its upstream cause cannot be reconstructed from stored bars.
    """
    if block.empty or list(block.feature) != repair.EXPECTED_FEATURES or block.shape[2] != 16:
        raise ValueError('15mraw Inf audit requires the rebuilt six-feature, 16-bar schema')
    evidence_path = folder / f'{evidence_name or proc.frame}_inf_evidence.jsonl'
    stats = item['inf_audit'] = {
        'status': 'running', 'evidence': str(evidence_path), 'inf_count': 0,
        'converted_to_nan': 0, 'by_feature': {}, 'by_origin': {}, 'by_mechanism': {},
        'action': 'Inf to NaN' if sanitize else 'audit before VWAP-to-close fallback',
        'examples': [], 'checked_dates': 0,
        'interpretation': '15mraw copies raw OHLCV and casts float64 to float32, then '
                          'uses close for zero-volume/Inf VWAP. It performs no price division. Reproduced VWAP mechanisms '
                          'are evidence from stored aggregates, not proof of vendor history.',
    }
    origins, mechanisms, features = Counter(), Counter(), Counter()
    seen = set()
    try:
        with evidence_path.open('w') as evidence:
            for d, date in enumerate(block.date):
                values = block.values[:, d]
                mask = torch.isinf(values)
                if vwap_only:
                    mask[..., block.feature != 'vwap'] = False
                positions = mask.nonzero().cpu().numpy()
                if len(positions):
                    source, source_path = load_source_day(int(date))
                    # Production stores canonical 0..15 minute labels. Do not
                    # guess positional mappings for malformed historical axes.
                    key = 'inday' if 'inday' in source.columns else 'minute'
                    if source.empty or 'secid' not in source or key not in source:
                        raise ValueError(f'Cannot trace Inf: missing source rows/keys at {source_path}')
                    labels = source[key].to_numpy()
                    if not np.isin(labels, np.arange(16)).all():
                        raise ValueError(f'Cannot trace noncanonical 15min bar labels at {source_path}')
                    source = source.drop_duplicates().set_index(['secid', key])
                    if source.index.has_duplicates:
                        raise ValueError(f'Cannot trace conflicting source rows at {source_path}')
                    for s, bar, f in positions:
                        secid, feature = int(block.secid[s]), str(block.feature[f])
                        if (secid, int(bar)) not in source.index or feature not in source:
                            raise ValueError(f'Cannot trace Inf: missing {secid}/{date}/{bar}/{feature} at {source_path}')
                        row = source.loc[(secid, int(bar))]
                        record = {'secid': secid, 'date': int(date), 'bar': int(bar),
                                  'feature': feature, 'source_path': source_path,
                                  **explain_inf(row, feature, float(values[s, bar, f]), str(source[feature].dtype))}
                        evidence.write(json.dumps(record, allow_nan=False) + '\n')
                        features[feature] += 1
                        origins[record['origin']] += 1
                        mechanisms[record['mechanism']] += 1
                        stats['inf_count'] += 1
                        # At most one log example per feature/origin/mechanism;
                        # complete cell-by-cell evidence remains on disk.
                        example_key = (feature, record['origin'], record['mechanism'])
                        if example_key not in seen:
                            seen.add(example_key)
                            stats['examples'].append(record)
                            Logger.note(f'15mraw/{proc.frame} Inf evidence: {json.dumps(record, ensure_ascii=False)}')
                    evidence.flush()
                    if sanitize:
                        values.masked_fill_(mask, torch.nan)
                        stats['converted_to_nan'] += len(positions)
                stats['checked_dates'] += 1
                if (d + 1) % 250 == 0:
                    Logger.note(f'15mraw/{proc.frame} Inf audit: {d + 1}/{len(block.date)} dates; '
                                f'{stats["inf_count"]} Inf traced')
        if origins['unexplained_preprocessing_inf']:
            raise ValueError('Unexplained finite/NaN-to-Inf preprocessing transition; see Inf evidence; not saving or training')
        stats['status'] = 'verified'
    except Exception:
        stats['status'] = 'failed'
        raise
    finally:
        stats.update(by_feature=dict(features), by_origin=dict(origins), by_mechanism=dict(mechanisms))
        Logger.note(f'15mraw/{proc.frame} Inf audit: {stats["status"]}; '
                    f'converted={stats["converted_to_nan"]}; origins={dict(origins)}; '
                    f'mechanisms={dict(mechanisms)}; evidence={evidence_path}')


def configure_processor(proc, item, folder):
    def observe(block):
        chunk = {'start': int(block.date[0]), 'end': int(block.date[-1])}
        item.setdefault('vwap_fallback_audits', []).append(chunk)
        audit_and_sanitize(block, proc, chunk, folder, vwap_only=True, sanitize=False,
                           evidence_name=f'{proc.frame}_vwap_{chunk["start"]}_{chunk["end"]}')
    proc.vwap_inf_observer = observe


def run_task(dry_run=False):
    plan = repair.repair_datablocks(dry_run=dry_run, prepare_block=audit_and_sanitize,
                                    configure_processor=configure_processor)
    if not dry_run:
        repair.request_training(REQUEST_ID)
    return plan


@ScriptTool('autofix_repair_15mraw_inf')
def main(dry_run: bool = False, **kwargs):
    run_task(dry_run=dry_run)


if __name__ == '__main__':
    main()
