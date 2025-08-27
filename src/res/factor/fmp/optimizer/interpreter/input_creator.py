import numpy as np
from typing import Any

from src.data import DATAVENDOR
from src.res.factor.util import Benchmark , Port , RISK_MODEL

from .bound import StockBound , StockPool , IndustryPool , GeneralBound , ValidRange , STOCK_UB , STOCK_LB
from .constr import LinearConstraint , TurnConstraint , CovConstraint , BoundConstraint , ShortConstraint


stock_bound_list : list[StockBound] = []
linear_bound_list : list[LinearConstraint] = []

def create_input_eq(opt_input : Any) -> float | Any:
    config = opt_input.cfg_equity
    if config['target_position']:  eq = config['target_position']
    elif config['target_value']:   eq = config['target_value'] / opt_input.initial_value
    elif config['add_position']:   eq = config['add_position'] + opt_input.initial_position
    elif config['add_value']:      eq = config['add_value'] / opt_input.initial_value + opt_input.initial_position
    return eq

def create_input_benchmark(opt_input : Any) -> np.ndarray | Any:
    bm_port = Benchmark.day_port(opt_input.benchmark  , opt_input.model_date , opt_input.cfg_benchmark['benchmark'])
    opt_input.benchmark_port = bm_port
    wb = bm_port.weight_align(opt_input.secid) * opt_input.eq if not bm_port.is_emtpy() else None
    return wb

def create_input_initial(opt_input : Any):
    pf : Port | Any = opt_input.initial_port
    w0 = pf.weight_align(opt_input.secid) if isinstance(pf , Port) and not pf.is_emtpy() else np.zeros(len(opt_input.secid))
    if (w0 == 0).all(): w0 = None
    return w0

def create_input_turn_con(opt_input : Any):
    max_turn = None if opt_input.initial_port is None else opt_input.cfg_turnover['double']
    return TurnConstraint(max_turn , opt_input.cfg_utility['trade_cost_rho'])

def create_input_cov_con(opt_input : Any):
    lmbd = opt_input.cfg_utility['lambda']
    te = opt_input.cfg_limitation['te_constraint']
    ignore_spec = opt_input.cfg_limitation['ignore_spec_risk']
    F , C , S = RISK_MODEL.get(opt_input.model_date).FCS_aligned(opt_input.secid)
    return CovConstraint(lmbd , te , F , C , None if ignore_spec else S , cov_type='model')

def create_input_bnd_con(opt_input : Any):

    # later ones has more priority
    append_bound_weight(opt_input)
    append_bound_limit(opt_input)
    append_bound_induspool(opt_input)
    append_bound_range(opt_input)
    append_bound_pool(opt_input)
    bound = StockBound.intersect_bounds(stock_bound_list , clear_after=True)
    assert not stock_bound_list , stock_bound_list

    if opt_input.cfg_short['short_position'] is None or opt_input.cfg_short['short_position'] <= 0: bound.update_lb(0)

    bnd_key = np.full(len(opt_input.secid) , 'ra')
    bnd_key[bound.ub == bound.lb] = 'fx'
    bnd_key[bound.ub >= STOCK_UB] = 'lo'
    bnd_key[bound.lb <= STOCK_LB] = 'up'

    return BoundConstraint(bnd_key , bound.lb , bound.ub)

def create_input_lin_con(opt_input : Any):

    append_linear_equity(opt_input) 
    append_linear_induspool(opt_input) 
    append_linear_board(opt_input) 
    append_linear_industry(opt_input) 
    append_linear_style(opt_input) 
    append_linear_component(opt_input)

    lin_con = LinearConstraint.stack(linear_bound_list , clear_after=True)
    assert not linear_bound_list , linear_bound_list

    return lin_con

def create_input_short_con(opt_input : Any):
    return ShortConstraint(opt_input.cfg_short['short_position'] , opt_input.cfg_short['short_cost'])

def append_bound_weight(opt_input : Any):
    bound_weight = StockBound.intersect_bounds([bnd.export(opt_input.wb) for bnd in opt_input.cfg_bound.values()])
    stock_bound_list.append(bound_weight)
    #return bound_weight

def append_bound_pool(opt_input : Any):
    pool : StockPool = opt_input.cfg_pool
    bound_pool = pool.export(opt_input.secid , opt_input.wb , opt_input.w0)
    stock_bound_list.append(bound_pool)

def append_bound_induspool(opt_input : Any):
    induspool : IndustryPool = opt_input.cfg_induspool
    bound_induspool = induspool.export(opt_input.w0 , RISK_MODEL.get(opt_input.model_date).industry(opt_input.secid))
    stock_bound_list.append(bound_induspool)

def append_bound_limit(opt_input : Any):
    w0 = 0 if opt_input.w0 is None else opt_input.w0
    secid : np.ndarray = opt_input.secid
    model_date : int = opt_input.model_date
    limitation : dict[str,Any] = opt_input.cfg_limitation
    bound_limit  = StockBound()
    if limitation.get('no_st'):
        df = DATAVENDOR.st_stocks
        pool = df[(df['entry_dt'] <= model_date) & (df['remove_dt'] >= model_date)]['secid'].to_numpy()
        bound_limit.intersect(StockPool.bnd_ub(secid , pool , 0))

    if ld := limitation.get('list_days'):
        df = DATAVENDOR.all_stocks
        pool = df[df['list_dt'] > DATAVENDOR.td(model_date , -ld).td]['secid'].to_numpy()
        bound_limit.intersect(StockPool.bnd_ub(secid , pool , 0))

    if limitation.get('kcb_no_buy'):
        pool = secid[(secid >= 688000) & (secid <= 689999)]
        bound_limit.intersect(StockPool.bnd_ub(secid , pool , w0))

    if limitation.get('kcb_no_sell'):
        pool = secid[(secid >= 688000) & (secid <= 689999)]
        bound_limit.intersect(StockPool.bnd_lb(secid , pool , w0))

    stock_bound_list.append(bound_limit)

def append_bound_range(opt_input : Any):
    secid : np.ndarray = opt_input.secid
    model_date : int = opt_input.model_date
    valid_ranges : dict[str,ValidRange] = opt_input.cfg_range
    bound_range = StockBound()
    for key , valid_range in valid_ranges.items():
        valname = key.split('.')[0] 
        if valname == 'ffmv' : 
            # value = DATAVENDOR.get_ffmv(secid , model_date)
            value = RISK_MODEL.get(model_date).weight(secid)
        elif valname == 'cp':
            value = DATAVENDOR.get_cp(secid , model_date)
        elif valname in ['amt' , 'bv']:
            print('Warning! [amt] and [bv] not yet define in [get_bound_range]!')
            continue
        else:
            raise KeyError(valname)
        assert isinstance(value , np.ndarray) , value
        bound_range.intersect(valid_range.export(value))

    stock_bound_list.append(bound_range)

def append_linear_equity(opt_input : Any):
    eq = np.array([opt_input.eq])
    eq_lin = LinearConstraint(np.ones((1,len(opt_input.secid))) , np.array(['fx']) , eq , eq)
    linear_bound_list.append(eq_lin)

def append_linear_induspool(opt_input : Any):
    induspool : IndustryPool = opt_input.cfg_induspool
    if not induspool.no_net_buy and not induspool.no_net_sell:  return

    industry = RISK_MODEL.get(opt_input.model_date).industry(opt_input.secid)
    w0 = np.zeros(len(industry)) if opt_input.w0 is None else opt_input.w0

    if induspool.no_net_buy:
        K = len(induspool.no_net_buy)
        A = np.stack([industry == ind for ind in induspool.no_net_buy] , axis=0)
        linear_bound_list.append(LinearConstraint(A , np.full(K , 'up') , np.full(K , -1.) , A.dot(w0)))

    if induspool.no_net_sell:
        K = len(induspool.no_net_sell)
        A = np.stack([industry == ind for ind in induspool.no_net_sell], axis=0)
        linear_bound_list.append(LinearConstraint(A , np.full(K , 'lo') , A.dot(w0) , np.full(K , 1.)))

def append_linear_board(opt_input : Any):
    board_bounds : dict[str,list[GeneralBound]] = opt_input.cfg_board
    if not board_bounds: return
    secid : np.ndarray = opt_input.secid
    model_date : int = opt_input.model_date
    for board_name , gen_bounds in board_bounds.items():
        if not gen_bounds: continue
        if board_name == 'shse':
            where = (secid >= 600000) * (secid <= 689999)
        elif board_name == 'szse':
            where = ((secid >= 0) * (secid <= 99999) + (secid >= 300000) * (secid <= 399999))
        elif board_name == 'bse':
            where = ((secid >= 200000) * (secid <= 299999) + (secid >= 800000) * (secid <= 899999))
        elif board_name == 'kcb':
            where = (secid >= 688000) * (secid <= 689999)
        elif board_name == 'csi':
            pool = np.concatenate([Benchmark('csi300').get(model_date,True).secid , 
                                   Benchmark('csi500').get(model_date,True).secid , 
                                   Benchmark('csi1000').get(model_date,True).secid])
            where = np.isin(secid , pool)
        else:
            raise KeyError(board_name)
        A , lin_type , lb , ub = gen_bounds[0].export_lin(1. * where , opt_input.wb , gen_bounds[1:])
        linear_bound_list.append(LinearConstraint(A , lin_type , lb , ub))

def append_linear_industry(opt_input : Any):
    indus_bounds : dict[str,list[GeneralBound]] = opt_input.cfg_industry
    if not indus_bounds: return
    industry = RISK_MODEL.get(opt_input.model_date).industry(opt_input.secid)
    for indus_name , gen_bounds in indus_bounds.items():
        if not gen_bounds: continue
        where = (industry == indus_name)
        A , lin_type , lb , ub = gen_bounds[0].export_lin(1. * where , opt_input.wb , gen_bounds[1:])
        linear_bound_list.append(LinearConstraint(A , lin_type , lb , ub))

def append_linear_style(opt_input : Any):
    style_bounds : dict[str,list[GeneralBound]] = opt_input.cfg_style
    if not style_bounds: return
    df = RISK_MODEL.get(opt_input.model_date).style(opt_input.secid)
    for style_name , gen_bounds in style_bounds.items():
        if not gen_bounds: continue
        value = df.loc[:,style_name].to_numpy()
        A , lin_type , lb , ub = gen_bounds[0].export_lin(value , opt_input.wb , gen_bounds[1:])
        linear_bound_list.append(LinearConstraint(A , lin_type , lb , ub))

def append_linear_component(opt_input : Any):
    comp_bounds : dict[str,list[GeneralBound]] = opt_input.cfg_component
    if not comp_bounds or opt_input.wb is None: return

    wb = opt_input.wb
    size = None
    for comp_name , gen_bounds in comp_bounds.items():
        if not gen_bounds: continue
        if comp_name == 'component':
            value = 1 * (opt_input.wb > 0)
        elif comp_name == 'bsizedev1':
            if size is None: size = RISK_MODEL.get(opt_input.model_date).style(opt_input.secid , 'size').to_numpy()
            value = np.abs(size - (size * wb / wb.sum()))
            value /= value.std()
        elif comp_name == 'bsizedev2':
            if size is None: size = RISK_MODEL.get(opt_input.model_date).style(opt_input.secid , 'size').to_numpy()
            value = np.square(size - (size * wb / wb.sum()))
            value /= value.std()
        else:
            raise KeyError(comp_name)
        A , lin_type , lb , ub = gen_bounds[0].export_lin(value , opt_input.wb , gen_bounds[1:])
        linear_bound_list.append(LinearConstraint(A , lin_type , lb , ub))