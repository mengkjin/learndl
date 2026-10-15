"""Source-qualified model labels, independent of config and data loading."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LabelSpec:
    source: str
    feature: str

    @property
    def name(self) -> str:
        return self.feature if self.source == 'y' else f'{self.source}@{self.feature}'

    @property
    def horizon(self) -> tuple[int, int] | None:
        """Return (start offset, holding days); open/VWAP imply T+1."""
        classic = re.fullmatch(r'(?:rtn|std|res)_lag(\d+)_(\d+)(?:_.*)?', self.feature)
        if classic:
            return int(classic[1]), int(classic[2])
        price = re.fullmatch(r'(?:rtn|std)_(\d+)_(?:open|vwap)', self.feature)
        return (1, int(price[1])) if price else None

    @classmethod
    def parse(cls, label: str) -> 'LabelSpec':
        if not isinstance(label, str) or label.count('@') > 1:
            raise ValueError(f'Invalid label reference: {label!r}')
        source, feature = label.split('@') if '@' in label else ('y', label)
        if not source or not feature or source != source.strip() or feature != feature.strip():
            raise ValueError(f'Invalid label reference: {label!r}')
        return cls(source, feature)


def parse_labels(labels: list[str]) -> list[LabelSpec]:
    if not isinstance(labels, list):
        raise ValueError(f'Labels must be a list of strings: {labels!r}')
    specs = [LabelSpec.parse(label) for label in labels]
    names = [spec.name for spec in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError(f'Labels must be nonempty and unique: {names}')
    return specs
