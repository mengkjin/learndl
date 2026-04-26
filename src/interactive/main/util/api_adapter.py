"""Streamlit UI for browsing and running ``src.api`` endpoints with ``[API Interaction]``."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import streamlit as st
import streamlit_antd_components as sac

from src.api.contract import APIEndpoint
from src.interactive.frontend import subheader_expander , ParamInputsForm
from src.interactive.main.util.script_detail import show_report_main , show_task_history
from src.interactive.main.util.session_control import SC

API_ADAPTER_SCRIPT_KEY = ".core/0.run_api_endpoint.py"

@st.cache_resource
def _cached_api_endpoint_list() -> stAPIEndpointList:
    """``(module, qual_path, endpoint_id)`` for exposed endpoints only."""
    return stAPIEndpointList(*(stAPIEndpoint(ep) for ep in APIEndpoint.iter_with_schema(exposed = True)))

class stAPIEndpoint(APIEndpoint):
    """API Endpoint for Streamlit API Console , including the properties of streamlit components"""
    def __init__(self , endpoint: APIEndpoint):
        super().__init__(endpoint.module , endpoint.qual_path , endpoint.func , endpoint.schema , endpoint.description , endpoint.parameters)
    
    @property
    def runner_key(self) -> str:
        """get the runner key for the api endpoint"""
        return f"{API_ADAPTER_SCRIPT_KEY}::{self.qualname}"

    @property
    def group_label(self) -> str:
        """get the group label for the api endpoint"""
        rest = self.module.removeprefix("src.api.").lstrip(".")
        if not rest:
            return "__group__/_root"
        return f"__group__/{rest.split(".")[0]}"

    @property
    def label(self) -> str:
        """get the label for the api endpoint"""
        return self.qualname

    @property
    def icon(self) -> str:
        """get the icon for the api endpoint"""
        return "file-earmark-code"

    @property
    def tree_tags(self) -> list:
        """get the tags for the api endpoint in the tree"""
        tags : list[sac.Tag] = []
        if self.email:
            tags.append(sac.Tag(label=f'Email' , color='blue'))
        return tags

    @property
    def detail_tags(self) -> list:
        """get the tags for the api endpoint in the detail"""
        tags : list[sac.Tag] = []
        risk_color = {
            "read_only": "green",
            "write": "yellow",
            "destructive": "red",
        }[self.risk]
        execution_color = {
            "immediate": "green",
            "short": "green",
            "medium": "yellow",
            "long": "red",
        }[self.execution_time]
        memory_color = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
        }[self.memory_usage]
        tags.append(sac.Tag(label=self.risk , color=risk_color))
        tags.append(sac.Tag(label=f'{self.execution_time} Execution' , color=execution_color))
        tags.append(sac.Tag(label=f'{self.memory_usage} Memory' , color=memory_color))
        tags.append(sac.Tag(label=", ".join(self.roles) , color='teal'))
        if self.lock_num >= 1:
            tags.append(sac.Tag(label=f'{self.lock_num} Locks' , color='orange'))
        if self.disable_platforms:
            tags.append(sac.Tag(label=f'{", ".join(self.disable_platforms)} Disabled' , color='gray'))
        return tags

class stAPIEndpointList:
    """List of API Endpoints for Streamlit API Console"""
    def __init__(self , *endpoints: stAPIEndpoint):
        self.endpoints = list(endpoints)
        self.endpoints = sorted(self.endpoints , key = lambda ep: ep.label)

    @classmethod
    def get_list(cls , needle: str | None = None) -> stAPIEndpointList:
        """get the api endpoint list from the cached api endpoint list, filter by needle if provided"""
        full_list = stAPIEndpointList(*_cached_api_endpoint_list())
        if needle:
            return full_list.filter(needle)
        return full_list

    def sac_tree_items(self) -> list:
        """get the sac (streamlit ant design) tree items for the api endpoint list"""
        grouped: dict[str, list[stAPIEndpoint]] = defaultdict(list)
        for ep in self.endpoints:
            grouped[ep.group_label].append(ep)
        items: list[dict[str, Any]] = []
        for group , members in grouped.items():
            children: list[dict[str, Any]] = []
            for ep in members:
                children.append({"label": ep.label , "icon": ep.icon , "tag": ep.tree_tags , 'size': 'xs'})
            items.append(
                {
                    "label": group ,
                    "icon": self.group_icon(group) ,
                    "children": children ,
                    "disabled": True ,
                }
            )
        self._sac_tree_items = items
        return self._sac_tree_items

    @classmethod
    def _preorder_tree_options(cls , items: list[dict[str, Any]]) -> list[str]:
        """get the preorder tree options for the api endpoint list"""
        labels: list[str] = []
        for it in items:
            labels.append(str(it.get("label" , "")))
            for ch in it.get("children") or []:
                if isinstance(ch , dict):
                    labels.extend(cls._preorder_tree_options([ch]))
        return labels

    def sac_tree_options(self) -> list[str]:
        """get the sac (streamlit ant design) tree options for the api endpoint list"""
        if not hasattr(self , "_sac_tree_options"):
            self.sac_tree_items()
        self._sac_tree_options = self._preorder_tree_options(self._sac_tree_items)
        return self._sac_tree_options

    @property
    def first_option(self) -> str | None:
        """get the first option for the api endpoint list (except the group), to be selected by default"""
        options = self.sac_tree_options()
        for option in options:
            if option and not option.startswith('__group__/'):
                return option
        return None

    def filter(self , needle: str) -> stAPIEndpointList:
        """filter the api endpoint list by needle"""
        return stAPIEndpointList(*(ep for ep in self.endpoints if needle.lower() in ep.label.lower()))

    def group_icon(self , group_label: str) -> str:
        """get the icon for the group label"""
        return "folder"

    def get_endpoint(self , option: str | None) -> stAPIEndpoint | None:
        """get the api endpoint by the option"""
        if not option:
            return None
        for ep in self.endpoints:
            if ep.label == option:
                return ep
        return None

    @classmethod
    def label_formatter(cls , label: str) -> str:
        """format the label for the api endpoint list"""
        if label.startswith("__group__/"):
            return label.removeprefix("__group__/").replace("_" , " ").title()
        return label.split(".")[-1]
    
    def __iter__(self):
        return iter(self.endpoints)

    def __len__(self):
        return len(self.endpoints)

    def __bool__(self):
        return bool(self.endpoints)

def _kwargs_dict_to_hex(d: dict[str, Any]) -> str:
    """Serialize call kwargs for ``--api_kwargs_hex`` (no spaces; safe for ScriptCmd)."""
    return json.dumps(d , separators = ("," , ":") , default = str).encode("utf-8").hex()

def show_api_browser() -> stAPIEndpoint | None:
    """show the api browser , including the tree and detail of selected api endpoint"""
    wkey = f'api-browser-{API_ADAPTER_SCRIPT_KEY}'
    header = 'API Browser'
    icon = ':material/code:'
    sub = subheader_expander(header , icon , True , help = "Browse and select API endpoints." , key = wkey)
    tree_key = "api_sac_tree"
    with sub:
        with st.container(height = 530):
            browser , detail = st.columns([2,3])
            with browser:
                needle = st.text_input("**_Filter Endpoints by module or name_**" , key = "api_console_search").strip().lower()
                endpoints = stAPIEndpointList.get_list(needle)

                if not endpoints:
                    st.info("No exposed API endpoints found (check ``expose: true`` in contracts).")
                    return

                tree_items = endpoints.sac_tree_items()
                tree_options = endpoints.sac_tree_options()
                first_option = endpoints.first_option
                if first_option is None:
                    st.warning("No selectable API leaves in tree.")
                    return

                option = SC.api_endpoint_selected
                if not isinstance(option , str) or option not in tree_options:
                    SC.api_endpoint_selected = first_option
                    option = first_option
                try:
                    default_index = tree_options.index(str(option))
                except ValueError:
                    SC.api_endpoint_selected = first_option
                    default_index = tree_options.index(first_option)

                valid_options = {ep.label for ep in endpoints}
                if tree_key in st.session_state:
                    stored_option = st.session_state[tree_key]
                    stored_option = stored_option if isinstance(stored_option , list) else (stored_option ,)
                    if any(v not in valid_options for v in stored_option if v is not None):
                        del st.session_state[tree_key]

                picked = sac.tree(
                    items = tree_items ,
                    index = default_index ,
                    format_func = stAPIEndpointList.label_formatter ,
                    open_all = True ,
                    show_line = True ,
                    height = 400 ,
                    return_index = False ,
                    key = tree_key ,
                )
                if picked:
                    option = picked[0] if isinstance(picked , list) else picked
                    if isinstance(option , str) and option:
                        if option != SC.api_endpoint_selected:
                            SC.api_endpoint_selected = option
                            st.rerun()

            with detail.container(gap = 'xxsmall'):
                option = SC.api_endpoint_selected
                if not option:
                    st.info("Select an API above.")
                    return

                endpoint = endpoints.get_endpoint(option)
                if endpoint is None:
                    st.error("Selected endpoint is missing or not exposed.")
                    return

                st.write(f'**{endpoint.qualname}**')

                if endpoint.schema:
                    sac.tags(endpoint.detail_tags , align='start' , format_func = lambda tag: tag.replace('_', ' ').title())                    

                if endpoint.description:
                    st.code(endpoint.description, language = 'markdown' , wrap_lines = True)

    return endpoint

def show_endpoint_parameters(endpoint: stAPIEndpoint | None = None) -> None:
    """show the parameters setting for the selected api endpoint"""
    if endpoint is None:
        endpoint = stAPIEndpointList.get_list().get_endpoint(SC.api_endpoint_selected)
    if endpoint is None:
        return
    
    param_inputs = endpoint.parameters
    empty_param = not param_inputs
    wkey = f'api-endpoint-param-setting-{endpoint.label}'
    header = 'Parameters Setting'
    icon = ':material/settings:'
    if empty_param:
        help = "Empty. No Parameter is Required for This Endpoint."
    else:
        help = "Input Parameters for This Endpoint in the Expanders Below, Mind the Required Ones."
    runner = SC.get_script_runner(API_ADAPTER_SCRIPT_KEY)
    subheader = subheader_expander(header , icon , True , help = help , key = wkey)
    
    with subheader:
        param_controls = st.empty()
        SC.param_inputs_form = ParamInputsForm.from_api_endpoint(endpoint , SC.script_params_cache , SC.get_task_item(SC.current_task_item)).init_param_inputs()
        assert isinstance(SC.param_inputs_form, ParamInputsForm) , "ParamInputsForm is not initiated"
        cols = param_controls.columns(6)
        with cols[0]:
            if st.button(":blue-badge[:material/refresh: **Reset Parameters**]", key = f"param-inputs-form-reset-param-button" , help = "Reset Parameters to Default" , type = 'tertiary'):
                SC.script_params_cache.clear_cache(endpoint.runner_key)
                SC.current_task_item = None
                SC.param_inputs_form.reset_options()

        with cols[1]:
            if st.button(":blue-badge[:material/history: **Last Parameters**]", key = f"param-inputs-form-last-param-button" , help = "Set Parameters to Latest Task's Parameters" , type = 'tertiary'):
                item = SC.get_latest_task_item(runner.script_key)
                if item is not None:
                    item_params = SC.param_inputs_form.cmd_to_param_values(cmd = item.cmd)
                    SC.script_params_cache.update_cache(endpoint.runner_key, 'value', item_params)
                    SC.param_inputs_form.reset_options()

    SC.refresh_run_button(runner)

def show_api_endpoint_console() -> None:
    """render api console"""
    show_task_history(API_ADAPTER_SCRIPT_KEY)
    show_api_browser()
    show_endpoint_parameters()
    show_report_main(API_ADAPTER_SCRIPT_KEY)
