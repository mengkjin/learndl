import torch , os

MODEL_DATATYPE_DICT = {
    'GRU'  : 'day' , # '15m'
    'LSTM' : 'day', 
    'TCN' : 'day', 
    'Transformer' : 'day' ,
    'GeneralRNN' : 'day' , # 'day' 
}

# model specifics
SHORTTEST   = False # True , False
NUM_MODELS  = 2
MODEL_MODULE    = 'GeneralRNN' # 'GeneralRNN'
MODEL_DATATYPE  = MODEL_DATATYPE_DICT[MODEL_MODULE]  #'day' , '15m' , 'gp' , 'day+15m'
MODEL_NICKNAME  = None # 'TCN_vs_LSTM' # any str or None

BATCH_SIZE = 10000
BEG_DATE , INPUT_SPAN , MAX_EPOCH = 20170103 , 2400 , 100

MODEL_PARAM = {
    'hidden_dim'    : [2 ** 5 , 2 ** 6], #  2**5
    'seqlens'       : [{'day' : 40, '15m' : 20 , 'dms' : 40}] , # although not a param for model, but can be different values(dict of SEQLEN)
    'rnn_layers'    : [4],
    'mlp_layers'    : [2], 
    'dropout'       : 0.1,
    'fc_in'         : [True],
    'fc_att'        : [True], 
    'type_rnn'      : ['tcn'], # 'gru' , 'lstm' , 'transformer'
    'rnn_att'       : [False],
    'ordered_param_group' : [False], # multiple data input , if train sequentially
    'num_output'    : [1],
    'hidden_as_factor'    : [False], 
    'kernel_size'   : [3] , 
    'ATF_mask'      : {'causal' : False , 'gaussian': False , 'tradegap': False ,} ,
}

TRAIN_PARAM = {
    'criterion':{
        'loss'      : 'ccc' , # mse , pearson , ccc
        'metric'    : 'pearson' , #mse,pearson,ccc,spearman
        'penalty'   : {'hidden_orthogonality' : 0.001 ,} ,  # 'hidden_orthogonality'
    },
    'trainer':{
        'optimizer' : {'name' : 'Adam' , 'param' : {} ,} , #{'name' : 'SGD' , 'param' : {} ,}
        'scheduler' : {'name' : 'cycle' , 'param' : {'base_lr':1e-7,'step_size_up':4}} ,
        'learn_rate': {
            'base'  : 0.005 , 
            'ratio' : {'attempt':[1 , 0.1 , 10 , 0.01 , 100] , 'round' : [1.] , 'transfer' : 0.1} ,
            'reset' : {'num_reset' : 2 , 'trigger' : 40 , 'recover_level' : 1. , 'speedup2x' : True , } , 
        } ,
        'gradient'  : {'clip_value' : 10.} ,
        'nanloss'   : {'retry' : 2} , 
        'retrain'   : {'attempts' : 4 , 'min_epoch' : 20 , 'min_epoch_round' : 10} , 
    },
    'transfer' : False ,
    'dataloader':{'train_ratio':0.85,'random_tv_split':True,'random_seed':42,},
    'terminate':{
        'overall' : {
            'early_stop'     : 10 , 'max_epoch'      : 100 , 
            'valid_converge' : {'min_epoch' : 5 , 'eps' : 1e-5} , #'train_converge' : {'min_epoch' : 5 , 'eps' : 1e-5} , 'tv_converge'    : {'min_epoch' : 5 , 'eps' : 1e-5} ,
        },
        'round' : {
            'early_stop'     : 5 , 'max_epoch'      : 50 , 
            'valid_converge' : {'min_epoch' : 5 , 'eps' : 1e-5} ,
        },
    },
    'multitask':{
        'type' : 'hybrid' , 
        'param_dict' : {
            'ewa':{},
            'hybrid':{'phi':None,'tau':2},
            'dwa':{'tau':2},
            'ruw':{'phi':None},
            'gls':{},
            'rws':{},
        },
    },
    'output_types': ['best' , 'swalast' , 'swabest'] , # ['best' , 'swalast' , 'swabest'] 
} 

COMPT_PARAM = {'cuda_first' : True, 'num_worker' : 10} # if not cuda_first, use parellel

# static variables (hardly change)
DDTYPE = torch.float # torch.double
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

END_DATE = 99991231 # 20170730 # 99991231
INTERVAL = 120 # model interval
INPUT_STEP_DAY = 5
TEST_STEP_DAY  = 1 # 1 , 5

VERBOSITY   = 2 # how much printout in training process (mostly) , 0~10, 10 will use tdqm
if SHORTTEST:
    BEG_DATE , INPUT_SPAN , MAX_EPOCH = 20221206 , 480 , 30
    VERBOSITY = 3 # print a little more
    MODEL_NICKNAME = 'SHORTTEST'

DATATYPE_TRADE  = ['day' , '30m' , '15m'] # 
DATATYPE_FACTOR = ['gp'  , 'dms']
DATATYPE_ORDER  = DATATYPE_TRADE + DATATYPE_FACTOR

TRAINER_TRANSFORMER = {
    'scheduler' : {'name' : 'cycle' , 'param' : {'base_lr':1e-7,'step_size_up':4} ,} ,
    'learn_rate': {
        'base'  : 0.005 , 
        'ratio' : {'attempt':[1 , 0.1 , 10 , 0.01 , 100] , 'round' : [1.] , 'transfer' : 0.1} ,
        'reset' : {'num_reset' : 1 , 'trigger' : 60 , 'recover_level' : 1. , 'speedup2x' : True , } , 
    } ,
    'gradient'  : {'clip_value' : 10.} ,
}

# types of scheduler:
# {'name' : 'step' , 'param' : {'step_size' : 10 , 'gamma' : 0.1}} 
# {'name':'cos','param':{'warmup_stage':10,'anneal_stage':40}}
# {'name' : 'cycle' , 'param' : {'min_lr':1e-7,'step_size_up':8,'step_size_down':2}} 

# check
if 'best' not in TRAIN_PARAM['output_types']: TRAIN_PARAM['output_types'] = ['best' , *TRAIN_PARAM['output_types']]
if 'num_output' not in MODEL_PARAM.keys(): MODEL_PARAM['num_output'] = [1]
if 'hidden_as_factor' not in MODEL_PARAM.keys(): MODEL_PARAM['hidden_as_factor'] = [False]