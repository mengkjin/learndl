"""In-memory source cleanup; never rewrites project database files."""
import pandas as pd

from .progress import progress


def deduplicate_table(frame: pd.DataFrame, keys: list[str], source: str,
                      audit: list[dict] | None = None) -> pd.DataFrame:
    duplicated = frame.duplicated(keys, keep=False)
    if not duplicated.any():
        return frame
    repeats = frame.loc[duplicated]
    distinct = repeats.drop_duplicates()
    conflicts = distinct.loc[distinct.duplicated(keys, keep=False), keys].drop_duplicates()
    removed = int(frame.duplicated(keys, keep='last').sum())
    event = {'source': source, 'keys': keys, 'removed_rows': removed,
             'duplicate_keys': len(repeats[keys].drop_duplicates()),
             'conflicting_keys': len(conflicts), 'keep': 'last_in_source_order',
             'sample': repeats[keys].drop_duplicates().head(5).to_dict('records')}
    if audit is not None:
        audit.append(event)
    progress('data', f'{source}: duplicate keys {keys}; removed {removed} rows, '
             f'conflicting keys={len(conflicts)}; keep last in source order '
             f'(not necessarily latest revision). Sample: {event["sample"]}', warning=True)
    return frame.drop_duplicates(keys, keep='last').copy()
