"""Causal data transforms, immutable bundles, model gradients and loader regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from src.data.preprocess.factor_sequence.schema import BuildConfig, EVENTS, KEYS
from src.data.preprocess.factor_sequence.numeric import stock_block, group_metrics, market_block, performance_block
from src.data.preprocess.factor_sequence.bundle import build, load, audit
from src.data.preprocess.factor_sequence.processors import load_block
from src.res.algo.nn.model.factor_sequence import MODELS
from src.res.model.util.data.operations import DataOperator
from src.res.model.util.data.module_data import ModuleData
from src.res.model.util.data.data_module import DataModule


def fixture(t=150):
    rng = np.random.default_rng(7)
    dates = np.arange(20200101,20200101+t)
    raw = rng.normal(size=(60,t,2))
    raw[0,:,0] = np.nan
    raw[1,3:8,1] = np.nan
    market = pd.DataFrame({'market_return':rng.normal(0,.01,t),
                           'atr':rng.uniform(.005,.04,t)},index=dates)
    for event in EVENTS:
        market[event] = 0.
    market.loc[dates[10],EVENTS[0]] = 1.
    return dict(raw=raw, returns=rng.normal(0,.01,(60,t)),dates=dates,
                secids=np.arange(60),market=market,
                events=pd.DataFrame([dict(event=EVENTS[0],event_date=dates[10],start=dates[8],end=dates[10])]))


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.cfg = BuildConfig(factors=('a','-b'))

    def test_mask_normalization_and_constant(self):
        data = fixture(15)
        data['raw'][:,2,1] = 9
        out,names,signed = stock_block(data['raw'],np.ones((60,15),bool),self.cfg)
        self.assertTrue(np.isfinite(out).all())
        k=2
        self.assertTrue((out[:,2,0,1] == 0).all())
        self.assertTrue((out[:,2,0,9*k+1] == 1).all())
        self.assertTrue((out[0,:,0,0] == 0).all())
        self.assertTrue((out[0,:,0,k] == 0).all())
        self.assertTrue((out[0,:,0,4*k] == 1).all())
        valid=out[:,3,0,k+1] > 0
        self.assertAlmostEqual(out[valid,3,0,1].mean(),0,places=6)
        self.assertAlmostEqual(out[valid,3,0,1].std(),1,places=6)
        # A newly reappearing observation describes the actual gap, not a zero-age update.
        self.assertGreater(out[1,8,0,3*k+1],out[1,2,0,3*k+1])
        self.assertTrue((out[...,7*k:9*k] == 0).all())

    def test_source_identifiers_are_required_per_file(self):
        from src.data.preprocess.factor_sequence.sources import read_daily_factor
        db=Mock()
        db.path.return_value.exists.return_value=True
        db.load.side_effect=[pd.DataFrame({'secid':[1,2],'date':[1,1],'a':[2,3]}),
                             pd.DataFrame({'date':[2,2],'a':[4,5]})]
        with self.assertRaisesRegex(ValueError,'row order'):
            read_daily_factor(db,'stock_factor','a',[1,2])

    def test_group_metrics_ties_and_missing(self):
        x=np.arange(60,dtype=float);r=x/100
        x[:5]=np.nan
        v=group_metrics(x,r,50)
        self.assertAlmostEqual(v[0],1)
        self.assertAlmostEqual(v[1],.245)
        self.assertTrue(np.isnan(group_metrics(np.ones(60),r,50)).all())
        self.assertTrue(np.isnan(group_metrics(x[:20],r[:20],50)).all())

    def test_maturity_and_unavailable_new_event(self):
        d=fixture(150)
        _,_,signed=stock_block(d['raw'],np.ones((60,150),bool),self.cfg)
        d['events']=pd.concat([d['events'],pd.DataFrame([dict(event=EVENTS[0],event_date=d['dates'][20],
                                                     start=d['dates'][0]-1,end=d['dates'][20])])])
        p,n=performance_block(signed,d['returns'],d['dates'],d['events'],self.cfg)
        self.assertEqual(p[0,9,0,n.index('a__ic_mean20__valid')],0)
        self.assertEqual(p[0,10,0,n.index('a__ic_mean20__valid')],1)
        self.assertEqual(p[0,60,0,n.index('a__top_risk_scaled__valid')],0)
        self.assertEqual(p[0,61,0,n.index('a__top_risk_scaled__valid')],1)
        self.assertEqual(p[0,20,0,n.index(EVENTS[0]+'__a__top__valid')],0)
        empty=d['events'].iloc[:0]
        with TemporaryDirectory() as tmp:
            path=build(tmp,**{**d,'events':empty},cfg=self.cfg)
            self.assertEqual(audit(path)['event_counts'],{})

    def test_prefix_and_confirmation(self):
        d=fixture(150)
        x,_,signed=stock_block(d['raw'],np.ones((60,150),bool),self.cfg)
        p,names=performance_block(signed,d['returns'],d['dates'],d['events'],self.cfg)
        idx=names.index(EVENTS[0]+'__a__top__valid')
        self.assertTrue((p[0,:10,0,idx] == 0).all())
        self.assertEqual(p[0,10,0,idx],1)
        prefix=d['events'].loc[d['events'].event_date <= d['dates'][79]]
        small,_,_=stock_block(d['raw'][:,:80],np.ones((60,80),bool),self.cfg)
        np.testing.assert_array_equal(small,x[:,:80])
        ps,_=performance_block(signed[:,:80],d['returns'][:,:80],d['dates'][:80],prefix,self.cfg)
        np.testing.assert_array_equal(ps,p[:,:80])
        m,_=market_block(d['market'],self.cfg)
        ms,_=market_block(d['market'].iloc[:80],self.cfg)
        np.testing.assert_array_equal(ms,m[:,:80])
        # History for z-score excludes the value currently being standardized.
        changed=d['market'].copy();changed.iloc[100,changed.columns.get_loc('atr')]=99
        mc,n=market_block(changed,self.cfg)
        self.assertEqual(mc[0,100,0,n.index('atr')],5)
        np.testing.assert_array_equal(mc[:,:100],m[:,:100])

    def test_bundle_version_corruption_and_chunks(self):
        d=fixture(85)
        with TemporaryDirectory() as tmp:
            a=build(Path(tmp)/'a',**d,cfg=self.cfg)
            b=build(Path(tmp)/'b',**d,cfg=self.cfg,chunk_size=30)
            for key in KEYS:
                np.testing.assert_array_equal(np.load(a/f'{key}.npz')['values'],np.load(b/f'{key}.npz')['values'])
            self.assertEqual(audit(a)['status'],'passed')
            self.assertRaises(ValueError,load,a,BuildConfig(factors=('a','b')))
            shared=load_block(a,'factorperf',d['dates'],np.arange(5000))
            self.assertEqual(shared.shape[0],1)
            from src.data.preprocess import PrePros
            processor=PrePros.get_processor('factorperf','fit',vb_level=0)
            self.assertFalse(processor.ENABLED)
            loaded=processor.load(d['dates'],secid=np.arange(5000),dataset=a)
            torch.testing.assert_close(loaded.values,shared.values)
            stocks=load_block(a,'factorseq',d['dates'],np.array([0,999]))
            self.assertTrue(torch.isfinite(stocks.values).all())
            self.assertTrue((stocks.values[1,...,8:10] == 1).all())
            with (a/'factorperf.npz').open('ab') as f:
                f.write(b'corruption')
            self.assertRaises(ValueError,load,a)

    def test_module_data_preserves_singleton_context_and_skips_norm(self):
        from src.data.util import DataBlock
        from src.data.preprocess.processors import PrePro_y
        from src.data.preprocess import PreProHistNorm
        d=fixture(25)
        with TemporaryDirectory() as tmp:
            path=build(tmp,**d,cfg=self.cfg)
            y=DataBlock(torch.randn(60,25,1,1),d['secids'],d['dates'],['std_lag1_10'])
            data=ModuleData.initialize(list(KEYS),['std_lag1_10'],factor_sequence_dataset=str(path),vb_level=0)
            with patch.object(PrePro_y,'load',return_value=y), \
                 patch('src.res.model.util.data.module_data.CALENDAR.range',return_value=d['dates']), \
                 patch.object(PreProHistNorm,'load_keys') as norms:
                data.load()
            norms.assert_not_called()
            self.assertEqual(list(data.x),list(KEYS))
            self.assertEqual(data.x['factorperf'].shape[0],1)
            self.assertEqual(data.x['factormarket'].shape[0],1)
            self.assertEqual(data.x['factorseq'].shape[0],60)
            self.assertFalse(data.config.fit_for_cache)

    def test_incremental_and_revision(self):
        d=fixture(100)
        with TemporaryDirectory() as tmp:
            prefix={**d,'raw':d['raw'][:,:80],'returns':d['returns'][:,:80],
                    'dates':d['dates'][:80],'market':d['market'].iloc[:80]}
            a=build(tmp,**prefix,cfg=self.cfg)
            b=build(tmp,**d,cfg=self.cfg,parent=a)
            for key in KEYS:
                np.testing.assert_array_equal(np.load(a/f'{key}.npz')['values'],np.load(b/f'{key}.npz')['values'][:,:80])
            d['raw'][2,3,0]+=1
            self.assertRaises(ValueError,build,tmp,**d,cfg=self.cfg,parent=a)
            c=build(tmp,**d,cfg=self.cfg,parent=a,allow_revision=True)
            self.assertTrue(load(c)['historical_revision'])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.x=[torch.randn(7,6,20),torch.randn(1,12,10),torch.randn(1,12,9)]

    def test_models_gradients_roundtrip_and_catalog(self):
        from src.res.algo.nn.api import get_nn_module
        for name,cls in MODELS.items():
            with self.subTest(name=name):
                self.assertIs(get_nn_module(name),cls)
                model=cls([20,10,9],hidden_dim=8,dropout=0)
                y,h=model(self.x)
                self.assertEqual(y.shape,(7,1));self.assertEqual(h['hidden'].shape,(7,8))
                loss=(y-torch.randn_like(y)).square().mean();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                saved=cls([20,10,9],hidden_dim=8,dropout=0)
                saved.load_state_dict(model.state_dict())
                torch.testing.assert_close(y,saved(self.x)[0])
                self.assertTrue(torch.isfinite(model([torch.zeros_like(t) for t in self.x])[0]).all())

    def test_identity_conditioning_and_causal_context(self):
        late=MODELS['factorseq_m1']([20,10,9],hidden_dim=8,dropout=0)
        cond=MODELS['factorseq_m2']([20,10,9],hidden_dim=8,dropout=0)
        cond.load_state_dict(late.state_dict(),strict=False)
        torch.testing.assert_close(late(self.x)[0],cond(self.x)[0])
        with torch.no_grad():
            cond.film.weight.normal_()
        changed=[v.clone() for v in self.x];changed[1][:,-1]+=20
        before=cond.performance(self.x[1])[0];after=cond.performance(changed[1])[0]
        torch.testing.assert_close(before[:,:-1],after[:,:-1])
        self.assertFalse(torch.allclose(cond(self.x)[0],cond(changed)[0]))
        self.assertRaises(ValueError,cond,[self.x[0],self.x[1].expand(7,-1,-1),self.x[2]])

    def test_missingness_ignores_values(self):
        model=MODELS['factorseq_mask']([20,10,9],hidden_dim=8,dropout=0)
        x=[v.clone() for v in self.x];x[0][...,:2]+=100;x[0][...,10:12]+=100
        torch.testing.assert_close(model(x)[0],model(self.x)[0])

    def test_purged_date_split(self):
        config=SimpleNamespace(sample_method='sequential',train_ratio=.8,batch_size=100,
                               Param={'train.dataloader.purge_horizon':3})
        proxy=SimpleNamespace(config=config,stage='fit')
        result=DataOperator.split_sample(proxy,torch.ones(3,20,dtype=torch.bool),torch.arange(3),torch.arange(20))
        train=torch.cat([b for b in result['train'] if b.numel()])
        valid=torch.cat(result['valid'])
        self.assertLess(int(train[:,1].max())+3,int(valid[:,1].min()))

    def test_effective_config_and_required_history(self):
        from src.data.preprocess.factor_sequence.pipeline import train_overrides, purged_split
        from src.res.model.util.config.config import ModelConfig
        split={'train':[20230103],'valid':[20240103],'test':[20250103]}
        config=ModelConfig(module='factorseq_m2',override=train_overrides('/tmp/fs_test','fs_offline_config',
                          'factorseq_m2',split,20250102),stage=0,resume=0,selection=0,vb_level=0)
        self.assertEqual(config.Param['train.factor_sequence.train_dates'],split['train'])
        self.assertEqual(config.Param['train.criterion.loss'],{'mse':{}})
        self.assertTrue(config.Param['input.factor_sequence.dataset'].endswith('/fs_test'))
        self.assertRaises(ValueError,purged_split,np.arange(100),np.arange(100)+11)
        result=purged_split(np.arange(700),np.arange(700)+11)
        self.assertLess(max(result['train'])+11,min(result['valid']))
        self.assertLess(max(result['valid'])+11,min(result['test']))

    def test_shared_batch_integrity_and_optimizer(self):
        from src.res.model.util.core.batch import BatchInput
        batch=BatchInput(self.x,torch.randn(7,1),None,torch.zeros(7,2,dtype=torch.long),
                         torch.ones(7,dtype=torch.bool),np.array([20240102]),np.arange(7))
        self.assertIs(batch.check_x_integrity_for_nn(),batch)
        for cls in MODELS.values():
            model=cls([20,10,9],hidden_dim=8,dropout=0)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            original=model.output.weight.detach().clone()
            for _ in range(3):
                optimizer.zero_grad()
                loss=(model(batch.x)[0]-batch.y).square().mean()
                loss.backward();optimizer.step()
            self.assertFalse(torch.equal(original,model.output.weight))
            with TemporaryDirectory() as tmp:
                torch.save(model.state_dict(),Path(tmp)/'model.pt')
                restored=cls([20,10,9],hidden_dim=8,dropout=0)
                restored.load_state_dict(torch.load(Path(tmp)/'model.pt',weights_only=True))
                torch.testing.assert_close(model(batch.x)[0],restored(batch.x)[0])

    def test_prediction_location_and_comparison_sample_guard(self):
        import json
        from src.data.preprocess.factor_sequence.pipeline import read_predictions, compare
        with TemporaryDirectory() as tmp:
            path=Path(tmp); preds=path/'snapshot/pred_recorder/preds';preds.mkdir(parents=True)
            frame=pd.DataFrame({'date':[20240102]*3,'secid':[1,2,3],'pred':[1.,2.,3.],
                                'label':[.1,.2,.3],'submodel':['best']*3})
            frame.to_feather(preds/'0.20240101.feather')
            self.assertEqual(len(read_predictions(path)),3)
            frame.to_csv(path/'one.csv',index=False);frame.iloc[:2].to_csv(path/'two.csv',index=False)
            runs=[dict(name=name,model='factorseq_mlp',seed=17,status='passed',seconds=1) for name in ('one','two')]
            (path/'runs.json').write_text(json.dumps(runs))
            np.savez(path/'realized_returns.npz',values=np.array([.1,.2,.3])[:,None,None,None],
                     secids=[1,2,3],dates=[20240102])
            with self.assertRaisesRegex(ValueError,'different samples'):
                compare(path)

    def test_shared_batch_extraction(self):
        config=SimpleNamespace(module_type='nn')
        class Operator:
            @staticmethod
            def rolling_rotation(key,data,i,j):
                return data[i[0]:i[0]+1,j[0]-2:j[0]+1,0]
        proxy=SimpleNamespace(config=config,data_operator=Operator())
        p=torch.randn(1,20,1,5)
        batch=DataModule.batch_data_x(proxy,{'factorperf':p},torch.arange(1000),torch.full((1000,),10))
        self.assertEqual(batch[0].shape,(1,3,5))
        self.assertRaises(ValueError,DataModule.batch_data_x,proxy,{'factorperf':p},torch.arange(2),torch.tensor([9,10]))

if __name__ == '__main__':
    unittest.main()
