"""Versioned, explicit schema; no database imports or implicit feature discovery."""
from dataclasses import asdict, dataclass
import hashlib
import json

KEYS = ('factorseq', 'factorperf', 'factormarket')
SHARED_KEYS = KEYS[1:]
VERSION = 1
FACTORS = (
    '-risk_lncap', 'risk_beta', 'btop', 'etop', 'stop', 'etop_est', 'epg_est',
    'sales_yoy', 'npro_yoy', 'sue_npro', 'sue_sales', 'outperform_npro',
    'roe_qtr', 'roa_qtr', 'roe_yoy', 'roa_yoy', 'analyst_recognition',
    'uppct_npro_3m', 'rec_npro_3m', 'outperform_titlepct', 'dtop',
    'pead_aog_rank', 'pead_alg_rank', '-turn_1m', '-turn_3m', 'ff_r2_1m',
    'atr_1m', '-mom_1m', '-mom_3m', 'umr_new_1m', 'umr_new_3m',
    'umr_new_6m', 'umr_new_12m',
)
EVENTS = ('selloff_rebound', 'platform_breakout', 'high_level_switch')
STOCK_CHANNELS = ('value', 'valid', 'observed', 'gap', 'never', 'change',
                  'change_valid', 'source_age', 'source_age_valid', 'degenerate')
DIAGNOSTIC_CHANNELS = ('valid', 'observed', 'gap', 'never', 'source_age', 'source_age_valid')

@dataclass(frozen=True)
class BuildConfig:
    factors: tuple[str, ...] = FACTORS
    min_cross_section: int = 20
    min_performance: int = 50
    lower_quantile: float = .01
    upper_quantile: float = .99
    history_window: int = 120
    history_min: int = 60
    age_cap: int = 252
    version: int = VERSION

    def __post_init__(self):
        if len(set(f.lstrip('+-') for f in self.factors)) != len(self.factors) or not self.factors:
            raise ValueError('Factors must be nonempty and unique')
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError('Invalid quantiles')
        if self.min_cross_section < 2 or self.min_performance < 10:
            raise ValueError('Insufficient cross-section limits')
        if not 1 <= self.history_min <= self.history_window or self.age_cap < 1:
            raise ValueError('Invalid history/age limits')
        if self.version != VERSION:
            raise ValueError('Unsupported schema version')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
