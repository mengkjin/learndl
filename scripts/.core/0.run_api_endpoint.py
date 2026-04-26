#! /usr/bin/env python3
# coding: utf-8
# author: jinmeng
# date: 2026-04-22
# description: Run Api Interaction
# content: Subprocess entry for Streamlit API console — resolves endpoint_id and runs through api-mode ScriptTool.
# email: False
# mode: shell

from typing import Any

from src.api.contract import APIEndpoint
from src.proj.util import ScriptTool

@ScriptTool("run_api_interaction" , "@qualname" , markdown_catcher = True)
def main(qualname: str = "" , **kwargs: Any) -> str:
    endpoint = APIEndpoint.from_qualname(qualname)
    out = endpoint.execute(**kwargs)
    return out


if __name__ == "__main__":
    main()
