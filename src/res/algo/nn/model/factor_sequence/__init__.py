"""Factor sequence baselines and context-conditioned models (M1/M2).

Streams: stock [N,60,Dx], performance [1,120,Dp], market [1,120,Dm].
Every batch is one date; shared streams are never copied to N in storage.
"""
from src.res.algo.registry import FACTOR_SEQUENCE_MODULES
import torch
from torch import nn

from .residual import FactorResidual
from .contextual_v4 import FactorContextualV4

CHANNELS = ('value', 'valid', 'observed', 'gap', 'never', 'change',
            'change_valid', 'source_age', 'source_age_valid', 'degenerate')
DIAGNOSTIC = ('valid', 'observed', 'gap', 'never', 'source_age', 'source_age_valid')


class FactorSequence(nn.Module):
    mode = 'm1'

    def __init__(self, input_dim, hidden_dim=32, dropout=.1, rnn_layers=2,
                 context_layers=1, modulation=True, use_performance=True,
                 use_market=True, gate_scale=.5, **kwargs):
        super().__init__()
        dims = [input_dim] if isinstance(input_dim, int) else list(input_dim)
        if len(dims) != 3 or dims[0] % len(CHANNELS):
            raise ValueError('Expected factorseq/factorperf/factormarket in that order')
        self.input_dim = dims
        self.factor_count = dims[0]//len(CHANNELS)
        self.use_performance, self.use_market = use_performance, use_market
        self.modulation = modulation
        if not 0 < gate_scale < 1:
            raise ValueError('gate_scale must keep the residual factor gate positive')
        self.gate_scale = gate_scale
        self.diagnostic_indices = [c*self.factor_count+k for c,n in enumerate(CHANNELS)
                                   if n in DIAGNOSTIC for k in range(self.factor_count)]
        if self.mode in ('mlp','mask'):
            width = dims[0] if self.mode == 'mlp' else len(self.diagnostic_indices)
            self.head = nn.Sequential(nn.Linear(width,hidden_dim),nn.GELU(),nn.Dropout(dropout))
        else:
            self.project = nn.Linear(dims[0],hidden_dim)
            self.stock = nn.GRU(hidden_dim,hidden_dim,rnn_layers,batch_first=True,
                                dropout=dropout if rnn_layers > 1 else 0)
            width = hidden_dim+self.factor_count
            if self.mode in ('m1','m2','gate'):
                self.performance = nn.GRU(dims[1],hidden_dim,context_layers,batch_first=True,
                                          dropout=dropout if context_layers > 1 else 0)
                self.market = nn.GRU(dims[2],hidden_dim,context_layers,batch_first=True,
                                     dropout=dropout if context_layers > 1 else 0)
                if self.mode != 'gate':
                    width += 2*hidden_dim
            if self.mode == 'm2':
                self.film = nn.Linear(2*hidden_dim,2*hidden_dim)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
            if self.mode == 'gate':
                gate_out = nn.Linear(hidden_dim, self.factor_count)
                self.factor_gate = nn.Sequential(nn.Linear(dims[0]+2*hidden_dim,hidden_dim),
                                                 nn.GELU(), gate_out)
                nn.init.zeros_(gate_out.weight)
                nn.init.zeros_(gate_out.bias)
            self.head = nn.Sequential(nn.Linear(width,hidden_dim),nn.GELU(),nn.Dropout(dropout))
        self.output = nn.Linear(hidden_dim,1)

    def gate_stock(self, stock, context):
        """Only factor levels are scaled; metadata and changes retain their semantics."""
        if context.shape[1] < stock.shape[1]:
            raise ValueError('Context window shorter than stock history')
        c = context[:,-stock.shape[1]:].expand(len(stock),-1,-1)
        gate = 1+self.gate_scale*torch.tanh(self.factor_gate(torch.cat((stock,c),dim=-1)))
        return torch.cat((stock[...,:self.factor_count]*gate,stock[...,self.factor_count:]),dim=-1),gate

    def forward(self, x):
        if not isinstance(x,(tuple,list)) or len(x) != 3:
            raise ValueError('Require three ordered streams')
        stock,p,m = [v.squeeze(2) if v.ndim == 4 and v.shape[2] == 1 else v for v in x]
        if any(v.ndim != 3 or v.shape[-1] != d for v,d in zip((stock,p,m),self.input_dim)):
            raise ValueError('Invalid stream dimensions')
        if p.shape[0] != 1 or m.shape[0] != 1 or p.shape[1] != m.shape[1]:
            raise ValueError('Context must be shared by a single-date batch with identical calendars')
        if self.mode == 'mlp':
            hidden = self.head(stock[:,-1])
        elif self.mode == 'mask':
            hidden = self.head(stock[:,-1,self.diagnostic_indices])
        else:
            z = self.project(stock)
            context = None
            if self.mode in ('m1','m2','gate'):
                ph = self.performance(p)[0]
                mh = self.market(m)[0]
                # A disabled branch contributes no bias/hidden-state information either.
                context = torch.cat((ph if self.use_performance else ph*0,
                                     mh if self.use_market else mh*0),dim=-1)
            if self.mode == 'gate' and self.modulation:
                gated,_ = self.gate_stock(stock,context)
                z = self.project(gated)
            if self.mode == 'm2' and self.modulation:
                if context is None:
                    raise RuntimeError('m2 modulation requires context')
                if context.shape[1] < stock.shape[1]:
                    raise ValueError('Context window shorter than stock history')
                gamma,beta = self.film(context[:,-stock.shape[1]:]).chunk(2,dim=-1)
                z = (1+gamma)*z+beta
            h = self.stock(z)[0][:,-1]
            parts = [h,stock[:,-1,:self.factor_count]]
            if context is not None and self.mode != 'gate':
                parts.append(context[:,-1].expand(len(stock),-1))
            hidden = self.head(torch.cat(parts,dim=-1))
        return self.output(hidden), {'hidden':hidden}


class FactorSnapshot(FactorSequence):
    mode = 'mlp'

class FactorGRU(FactorSequence):
    mode = 'gru'

class FactorLateFusion(FactorSequence):
    mode = 'm1'

class FactorConditioned(FactorSequence):
    mode = 'm2'

class FactorMissingness(FactorSequence):
    mode = 'mask'

class FactorValueGate(FactorSequence):
    """Stock-conditioned factor gates; context has no direct prediction-head path."""
    mode = 'gate'



MODELS = {name: globals()[class_name] for name, class_name in FACTOR_SEQUENCE_MODULES.items()}
CATALOG = {name: {'status':'implemented','version':1,'mode':model.mode,
                  'streams':['factorseq','factorperf','factormarket'],
                  'windows':[60,120,120], 'shared_streams':['factorperf','factormarket'],
                  'defaults':{'hidden_dim':32,'rnn_layers':2,'context_layers':1,'dropout':.1},
                  'config':f'configs/algo/nn/{name}.yaml',
                  'experiment':'configs/experience/model/factor_sequence/README.md'}
           for name,model in MODELS.items()}
CATALOG['factorseq_residual'].update(version=2, channels=['value','valid','gap','never'])
CATALOG['factorseq_residual']['variants'] = {
    'contextual_v4': {
        'version': 4,
        'status': 'legacy',
        'streams': ['factorseq', 'factorperf'],
        'windows': [60, 120],
        'shared_streams': ['factorperf'],
        'gate_target': 'current linear factor contributions',
        'gate_hidden_dim': 8,
        'gate_half_lives': [5.0, 20.0, 60.0],
        'gate_scale': 0.2,
        'gate_min_observations': 20,
        'gate_penalty': 0.01,
        'schedule': 'configs/schedule/archive/factorseq_v4_contextual.yaml',
        'experiment': 'configs/experience/model/factor_sequence/contextual_v4.md',
    }
}
for _suffix, _parameters in [('market', 22506), ('events', 22551)]:
    CATALOG['factorseq_residual']['variants'][f'contextual_v4_{_suffix}'] = {
        **CATALOG['factorseq_residual']['variants']['contextual_v4'],
        'streams': ['factorseq', 'factorperf', 'factormarket'],
        'windows': [60, 120, 1],
        'shared_streams': ['factorperf', 'factormarket'],
        'market_rank': 2,
        'parameters': _parameters,
        'context_parameters': _parameters - 22275,
        'schedule': f'configs/schedule/archive/factorseq_v4_contextual_{_suffix}.yaml',
        'experiment': 'configs/experience/model/factor_sequence/contextual_v4_market.md',
    }
CATALOG['factorseq_residual']['variants']['contextual_v4_events'].update(
    event_rank=1, event_decay_days=20.0)
CATALOG['factorseq_contextual_v4'].update(
    version=4, mode='shared', streams=['factorseq', 'factorperf'],
    windows=[60, 60], shared_streams=['factorperf'],
    defaults={'hidden_dim':32,'rnn_layers':2,'dropout':.1,'context_dropout':.1,
              'min_observations':20,'context_penalty':.01},
    variants={variant: {'parameters': count, 'schedule': f'factorseq_v4_{variant}'}
              for variant, count in [('shared', 23979), ('private', 28003), ('graph', 24975)]},
    experiment='configs/experience/model/factor_sequence/contextual_v4_sequence.md')
CATALOG['factorseq_gate'].update(gate_target='factor levels only',
    gate_inputs='stock values and metadata + causal performance/market context',
    context_to_head=False,gate_scale=.5)
CATALOG.update({name:{'status':'not_implemented','version':None,'dependency':dependency}
                for name,dependency in [('M3','Dated event/factor memory and causal attention'),
                                         ('M4','Reliable update timestamps or explicit weekly sampling'),
                                         ('M5','Point-in-time sparse graph and single-date stock batches')]})
