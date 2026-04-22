"""CI / local checks for ``src.api`` ``[API Interaction]`` blocks and bind helpers."""
from __future__ import annotations

import inspect
import unittest

from src.api.contract import (
    assert_all_api_contracts_ok ,
    bind_explicit_only ,
    describe_api_callable ,
    explicit_signature_parameters ,
    filter_kwargs_explicit_only ,
    interaction_for_callable ,
    parse_interaction_block ,
    validate_interaction_schema ,
)
from src.api.data import DataAPI
from src.api.trading import TradingAPI


class TestParseInteraction(unittest.TestCase):
    def test_parse_block(self) -> None:
        doc = '''Summary.

        [API Interaction]:
          expose: true
          roles: [user]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        '''
        d = parse_interaction_block(doc)
        self.assertIsNotNone(d)
        assert d is not None
        self.assertTrue(d["expose"])
        self.assertEqual(d["lock_num"] , -1)

    def test_missing_header(self) -> None:
        self.assertIsNone(parse_interaction_block("no block here"))


class TestValidateSchema(unittest.TestCase):
    def test_forbid_max_concurrent(self) -> None:
        d = {
            "expose": True ,
            "roles": ["developer"] ,
            "risk": "read_only" ,
            "lock_num": 1 ,
            "disable_platforms": [] ,
            "execution_time": "immediate" ,
            "memory_usage": "low" ,
            "max_concurrent": 2 ,
        }
        errs = validate_interaction_schema(d)
        self.assertTrue(any("max_concurrent" in e for e in errs))

    def test_override_arg_attr_nested(self) -> None:
        d = {
            "expose": True ,
            "roles": ["developer"] ,
            "risk": "read_only" ,
            "lock_num": 1 ,
            "disable_platforms": [] ,
            "execution_time": "short" ,
            "memory_usage": "medium" ,
            "override_arg_attr": {"benchmark": {"type": "str" , "default": "defaults"}} ,
        }
        self.assertEqual(validate_interaction_schema(d) , [])


class TestBindExplicit(unittest.TestCase):
    def test_trading_analyze_drops_var_keyword(self) -> None:
        sig = inspect.signature(TradingAPI.analyze)
        names = set(explicit_signature_parameters(sig))
        self.assertIn("port_name" , names)
        self.assertNotIn("kwargs" , names)
        user = {"port_name": "p" , "start": 20200101 , "extra_unknown": 1 , "foo": 2}
        filt = filter_kwargs_explicit_only(sig , user)
        self.assertEqual(set(filt.keys()) , {"port_name" , "start"})
        ba = bind_explicit_only(TradingAPI.analyze , user)
        self.assertIn("port_name" , ba.arguments)

    def test_data_is_updated_contract(self) -> None:
        d = interaction_for_callable(DataAPI.is_updated)
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(validate_interaction_schema(d) , [])


class TestAllApiContracts(unittest.TestCase):
    def test_full_package_schema(self) -> None:
        assert_all_api_contracts_ok()


class TestDescribeApiCallable(unittest.TestCase):
    def test_trading_available_ports_describe(self) -> None:
        d = describe_api_callable(TradingAPI.available_ports)
        self.assertIn('Const.TradingPort' , d['description'])
        self.assertTrue(d['schema']['expose'])
        self.assertEqual(len(d['parameters']) , 1)
        self.assertEqual(d['parameters'][0]['name'] , 'backtest')


if __name__ == "__main__":
    unittest.main()
