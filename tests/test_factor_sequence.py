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
        old = pd.DataFrame([dict(event=EVENTS[0],event_date=d['dates'][0]-i,
                                start=d['dates'][0]-i-2,end=d['dates'][0]-i)
                            for i in (10,20,30)])
        d['events']=pd.concat([old,d['events']],ignore_index=True)
        with TemporaryDirectory() as tmp:
            a=build(Path(tmp)/'a',**d,cfg=self.cfg)
            b=build(Path(tmp)/'b',**d,cfg=self.cfg,chunk_size=30)
            for key in KEYS:
                np.testing.assert_array_equal(np.load(a/f'{key}.npz')['values'],np.load(b/f'{key}.npz')['values'])
            # Accessing a compressed NPZ member repeatedly decodes the complete
            # panel each time. Keep audit IO bounded as date/factor counts grow.
            with np.load(a/'inputs.npz') as archive:
                archive_type = type(archive)
            original_getitem = archive_type.__getitem__
            reads = {}
            def count_reads(archive, key):
                reads[key] = reads.get(key,0)+1
                return original_getitem(archive,key)
            with patch.object(archive_type,'__getitem__',count_reads):
                report=audit(a)
                self.assertEqual(report['status'],'passed')
                self.assertEqual(len(report['event_checks']),2)
            self.assertEqual(reads['raw'],1)
            self.assertEqual(reads['values'],3)
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
            with patch.object(PrePro_y,'load',autospec=True,return_value=y), \
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

class ServerSuiteTests(unittest.TestCase):
    def test_cli_discovery_and_callable_entrypoint(self):
        import sys
        from src.api.util.backend.script import ScriptRunner, iter_runnable_scripts
        from src.api.calls.script import RunPipelineScript
        from src.proj.util.script.param_schema import ScriptParamSchema
        key='4_train/6_factor_sequence_verify.py'
        self.assertIn(key,[runner.script_key for runner in iter_runnable_scripts()])
        runner=ScriptRunner.from_key(key)
        self.assertFalse(runner.header.disabled)
        self.assertFalse(runner.header.email)  # The suite owns mail, no second notification.
        main=RunPipelineScript._load_main(runner)
        schema=ScriptParamSchema.from_script(runner.script,main=main)
        self.assertEqual(schema.sig_default('test_start'),20170101)
        self.assertEqual({p.name for p in schema.params},set(schema.signature_defaults))
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch('src.data.preprocess.factor_sequence.server_suite.run_suite',
                       return_value=(folder,{'status':'passed'})) as run, \
                 patch('src.data.preprocess.factor_sequence.server_suite.send_result',
                       return_value={'status':'sent'}) as mail, \
                 patch.object(sys,'argv',['host-cli','--unrelated-host-option']):
                self.assertEqual(main(run_id='cli_mock',test_start=20170101,research=True),0)
                self.assertEqual(run.call_args.args[2],'cli_mock')
                self.assertTrue(run.call_args.args[6])
                mail.assert_called_once_with(folder,None)
                run.return_value=(folder,{'status':'failed'})
                with self.assertRaisesRegex(RuntimeError,'verification failed'):
                    main(run_id='cli_mock_failed')
                self.assertEqual(mail.call_count,2)


    def test_gate_identity_bounds_and_causality(self):
        torch.manual_seed(13)
        base=MODELS['factorseq_gru']([20,10,9],hidden_dim=8,dropout=0)
        gate=MODELS['factorseq_gate']([20,10,9],hidden_dim=8,dropout=0)
        gate.load_state_dict(base.state_dict(),strict=False)
        x=[torch.randn(7,6,20),torch.randn(1,12,10),torch.randn(1,12,9)]
        torch.testing.assert_close(base(x)[0],gate(x)[0])
        with torch.no_grad():
            gate.factor_gate[-1].weight.normal_()
        c=torch.cat((gate.performance(x[1])[0],gate.market(x[2])[0]),-1)
        z,g=gate.gate_stock(x[0],c)
        torch.testing.assert_close(z[...,2:],x[0][...,2:])
        self.assertTrue(((g>=.5)&(g<=1.5)).all())
        self.assertFalse(torch.allclose(g[0],g[1]))
        changed=x[1].clone();changed[:,-1]+=30
        c2=torch.cat((gate.performance(changed)[0],gate.market(x[2])[0]),-1)
        z2,_=gate.gate_stock(x[0],c2)
        torch.testing.assert_close(z[:,:-1],z2[:,:-1])
        self.assertFalse(torch.allclose(z[:,-1],z2[:,-1]))

    def test_discovery_uses_earliest_history_without_stale_factor_cutoff(self):
        from src.data.preprocess.factor_sequence.server_suite import discover
        report=dict(status='passed',missing=[],factors={
            'a':dict(start=20050104,end=20200102),'b':dict(start=20150105,end=20261029)})
        with patch('src.data.preprocess.factor_sequence.sources.preflight',return_value=report), \
             patch('src.proj.DB') as db, patch('src.proj.CALENDAR') as calendar:
            # Dates exposes max as a property, not numpy's max() method.
            from unittest.mock import MagicMock
            dates=MagicMock();dates.__len__.return_value=2;dates.max=20261029
            db.dates.return_value=dates;calendar.updated.return_value=20261030
            actual=discover(BuildConfig(factors=('a','b')))
        self.assertEqual(actual['history_start'],20050104)
        self.assertEqual(actual['data_end'],20261029)

    def test_walk_forward_exact_coverage_and_purge(self):
        from src.data.preprocess.factor_sequence.pipeline import walk_forward_splits
        dates=np.arange(1000);ends=dates+11
        folds=walk_forward_splits(dates,ends,600,750)
        self.assertEqual([d for f in folds for d in f['test']],list(range(600,751)))
        self.assertEqual([len(f['test']) for f in folds],[63,63,25])
        for f in folds:
            self.assertLess(max(f['train'])+11,min(f['valid']))
            self.assertLess(max(f['valid'])+11,min(f['test']))
        with self.assertRaisesRegex(ValueError,'Insufficient history'):
            walk_forward_splits(dates,ends,300)

    def test_aggregate_and_reject_missing_dates(self):
        from src.data.preprocess.factor_sequence.server_suite import aggregate
        from src.data.preprocess.factor_sequence.bundle import write_json
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folders=[]
            for idx,ics in enumerate(([.1,.2],[.5])):
                folder=root/str(idx);training=folder/'training';training.mkdir(parents=True)
                write_json(training/'runs.json',[dict(name='m',model='factorseq_gate',seed=17,status='passed')])
                pd.DataFrame(dict(date=np.arange(len(ics))+idx*2,rank_ic=ics,
                    top_decile_10d_excess=ics,n=[60]*len(ics))).to_csv(training/'m_daily.csv',index=False)
                folders.append(folder)
            result=aggregate(folders,root,[0,1,2])[0]
            self.assertAlmostEqual(result['icir'],np.mean([.1,.2,.5])/np.std([.1,.2,.5]))
            with self.assertRaisesRegex(ValueError,'gaps/overlap'):
                aggregate(folders,root,[0,1,2,3])

    def test_suite_failure_is_recorded_and_stops(self):
        from src.data.preprocess.factor_sequence import server_suite as suite
        with TemporaryDirectory() as tmp:
            with patch.object(suite,'execute',side_effect=RuntimeError('offline failed')) as execute:
                folder,report=suite.run_suite(Path(__file__).resolve().parents[1],tmp,'failed',
                    Path(__file__).resolve().parents[1]/'configs/experience/model/factor_sequence/data.json')
            self.assertEqual(execute.call_count,1)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['stages'][0]['status'],'failed')
            self.assertTrue((folder/'suite.json').exists())

    def test_suite_success_orders_stages_and_freezes_range(self):
        import json
        from src.data.preprocess.factor_sequence import server_suite as suite
        from src.data.preprocess.factor_sequence.bundle import write_json
        with TemporaryDirectory() as tmp:
            root=Path(__file__).resolve().parents[1];results=Path(tmp)
            dataset=results/'data';dataset.mkdir()
            def child(command,log,workdir):
                Path(log).write_text('mocked child; no real training')
                if 'build' in command:
                    write_json(results/'success/dataset.json',{'path':str(dataset)})
                    self.assertEqual(command[command.index('--start')+1],'20050104')
                if 'compare' in command:
                    run_id=command[command.index('--run-id')+1]
                    training=results/run_id/'training';training.mkdir(parents=True)
                    write_json(training/'runs.json',[dict(name='m',model='factorseq_gate',seed=17,status='passed')])
                    pd.DataFrame(dict(date=[20170103],rank_ic=[.1],top_decile_10d_excess=[.02],n=[60])).to_csv(
                        training/'m_daily.csv',index=False)
            plan=dict(latest_mature_label=20170103,folds=[dict(train=[20150105],valid=[20161003],test=[20170103])])
            with patch.object(suite,'execute',side_effect=child), \
                 patch.object(suite,'discover',return_value=dict(status='passed',history_start=20050104,data_end=20170120)), \
                 patch.object(suite,'load',return_value={'version':'test'}), \
                 patch.object(suite,'available_splits',return_value=plan) as split:
                folder,report=suite.run_suite(root,results,'success',root/'configs/experience/model/factor_sequence/data.json')
            self.assertEqual(report['status'],'passed')
            self.assertEqual([v['name'] for v in report['stages']],
                ['offline','discover','build','audit','plan','smoke_000','compare_000','compare_all'])
            split.assert_called_once_with(dataset,20170101,None)
            self.assertEqual(json.loads((folder/'suite.json').read_text())['actual_test_start'],20170103)

    def test_mail_packaging_without_sending(self):
        from src.data.preprocess.factor_sequence import server_suite as suite
        from src.data.preprocess.factor_sequence.bundle import write_json
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write_json(folder/'suite.json',dict(status='failed',run_id='test',test_start=20170101,
                stages=[dict(name='audit',status='failed')],fold_directories=[],interpretation='verification'))
            with patch('src.proj.MACHINE') as machine, patch('src.proj.util.web.task_mail.TaskResultMail') as mail:
                machine.emailable=False
                self.assertEqual(suite.send_result(folder)['status'],'skipped')
                mail.assert_not_called()
                machine.emailable=True;mail.return_value.return_value=True
                self.assertEqual(suite.send_result(folder,'test@example.invalid')['status'],'sent')
                self.assertEqual(mail.return_value.call_args.kwargs['recipient'],'test@example.invalid')
                self.assertTrue((folder/'result_report.zip').exists())


if __name__ == '__main__':
    unittest.main()
