"""Parse and validate ``[API Interaction]:`` docstring blocks for ``src.api`` endpoints.

Provides YAML extraction, schema checks (locks, enums, no ``max_concurrent``),
discovery over the ``src.api`` package, and helpers to bind call arguments while
ignoring ``*args`` / ``**kwargs`` buckets.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any , Callable , Iterator

import yaml

INTERACTION_HEADER = "[API Interaction]:"

ROLES = frozenset({"user" , "developer" , "admin"})
RISKS = frozenset({"read_only" , "write" , "destructive"})
PLATFORMS = frozenset({"windows" , "linux" , "macos"})
EXECUTION_TIMES = frozenset({"immediate" , "short" , "medium" , "long"})
MEMORY_USAGE = frozenset({"low" , "medium" , "high"})

# Contract keys (``max_concurrent`` is forbidden).
REQUIRED_INTERACTION_KEYS = frozenset({
    "expose" , "roles" , "risk" , "lock_num" , "disable_platforms" ,
    "execution_time" , "memory_usage" ,
})
OPTIONAL_INTERACTION_KEYS = frozenset({
    "lock_timeout" , "override_arg_attr" , "email" ,
})
KNOWN_INTERACTION_KEYS = REQUIRED_INTERACTION_KEYS | OPTIONAL_INTERACTION_KEYS
FORBIDDEN_INTERACTION_KEYS = frozenset({"max_concurrent"})


def parse_interaction_block(doc: str | None) -> dict[str, Any] | None:
    """Return parsed mapping under ``[API Interaction]:``, or None if absent or empty."""
    if not doc or INTERACTION_HEADER not in doc:
        return None
    tail = doc.split(INTERACTION_HEADER , 1)[1]
    lines = tail.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    widths = [len(line) - len(line.lstrip(" \t")) for line in lines if line.strip()]
    if not widths:
        return None
    margin = min(widths)
    dedented = "\n".join(line[margin:] if len(line) >= margin else line for line in lines)
    try:
        data = yaml.safe_load(dedented)
    except yaml.YAMLError:
        return None
    if data is None:
        return None
    if not isinstance(data , dict):
        return None
    return data


def _err(msg : str) -> str:
    return msg


def validate_interaction_schema(
    data : dict[str, Any] ,
    *,
    strict_unknown_keys : bool = True ,
) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errs : list[str] = []

    for forbidden in FORBIDDEN_INTERACTION_KEYS:
        if forbidden in data:
            errs.append(_err(f"forbidden key {forbidden!r} (use lock_num / lock_timeout instead)"))

    if strict_unknown_keys:
        for k in data:
            if k not in KNOWN_INTERACTION_KEYS:
                errs.append(_err(f"unknown key {k!r}"))

    missing = REQUIRED_INTERACTION_KEYS - data.keys()
    if missing:
        errs.append(_err(f"missing required keys: {sorted(missing)!r}"))

    if "expose" in data and not isinstance(data["expose"] , bool):
        errs.append(_err("expose must be a boolean"))

    roles = data.get("roles")
    if roles is not None:
        if not isinstance(roles , list) or not roles:
            errs.append(_err("roles must be a non-empty list"))
        else:
            bad = [r for r in roles if r not in ROLES]
            if bad:
                errs.append(_err(f"invalid roles {bad!r}; allowed {sorted(ROLES)!r}"))

    if "risk" in data and data["risk"] not in RISKS:
        errs.append(_err(f"risk must be one of {sorted(RISKS)!r}"))

    if "lock_num" in data and not isinstance(data["lock_num"] , int):
        errs.append(_err("lock_num must be an integer"))

    if "lock_timeout" in data and data["lock_timeout"] is not None:
        if not isinstance(data["lock_timeout"] , int):
            errs.append(_err("lock_timeout must be an integer when present"))

    dpf = data.get("disable_platforms")
    if dpf is not None:
        if not isinstance(dpf , list):
            errs.append(_err("disable_platforms must be a list"))
        else:
            bad = [p for p in dpf if p not in PLATFORMS]
            if bad:
                errs.append(_err(f"invalid disable_platforms entries {bad!r}"))

    et = data.get("execution_time")
    if et is not None and et not in EXECUTION_TIMES:
        errs.append(_err(f"execution_time must be one of {sorted(EXECUTION_TIMES)!r}"))

    mu = data.get("memory_usage")
    if mu is not None and mu not in MEMORY_USAGE:
        errs.append(_err(f"memory_usage must be one of {sorted(MEMORY_USAGE)!r}"))

    if "email" in data and data["email"] is not None and not isinstance(data["email"] , bool):
        errs.append(_err("email must be a boolean when present"))

    oaa = data.get("override_arg_attr")
    if oaa is not None:
        if not isinstance(oaa , dict):
            errs.append(_err("override_arg_attr must be a mapping (param name -> attr dict)"))
        else:
            for pname , spec in oaa.items():
                if not isinstance(pname , str) or not pname:
                    errs.append(_err("override_arg_attr keys must be non-empty strings"))
                    continue
                if not isinstance(spec , dict):
                    errs.append(_err(f"override_arg_attr[{pname!r}] must be a mapping"))
                    continue
                for ik , iv in spec.items():
                    if not isinstance(ik , str):
                        errs.append(_err(f"override_arg_attr[{pname!r}] has non-string key {ik!r}"))
                    elif iv is not None and not isinstance(iv , (str , int , float , bool , list)):
                        errs.append(
                            _err(
                                f"override_arg_attr[{pname!r}][{ik!r}] must be scalar or list, "
                                f"not {type(iv).__name__}"
                            )
                        )

    return errs


def interaction_for_callable(obj : Any) -> dict[str, Any] | None:
    """Parse ``[API Interaction]`` for a function or method-like *obj*."""
    doc = inspect.getdoc(obj)
    return parse_interaction_block(doc)


def explicit_signature_parameters(sig : inspect.Signature) -> dict[str , inspect.Parameter]:
    """Drop VAR_POSITIONAL and VAR_KEYWORD parameters (``*args`` / ``**kwargs``)."""
    return {
        n: p
        for n , p in sig.parameters.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL , inspect.Parameter.VAR_KEYWORD)
    }


def filter_kwargs_explicit_only(sig : inspect.Signature , kwargs : dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that map to explicit (non var-*) parameters."""
    explicit = explicit_signature_parameters(sig)
    return {k: v for k , v in kwargs.items() if k in explicit}


def bind_explicit_only(callable_obj : Callable[..., Any] , kwargs : dict[str, Any]) -> inspect.BoundArguments:
    """Bind *kwargs* to *callable_obj* using only explicit parameters (exclude ``**kwargs`` / ``*args``)."""
    sig = inspect.signature(callable_obj)
    filtered = filter_kwargs_explicit_only(sig , kwargs)
    return sig.bind_partial(**filtered)


def _walk_class_attrs(cls : type , prefix : str) -> Iterator[tuple[str , Callable[..., Any]]]:
    for key , value in cls.__dict__.items():
        if key.startswith("_"):
            continue
        if isinstance(value , type):
            yield from _walk_class_attrs(value , f"{prefix}.{key}")
        elif isinstance(value , classmethod):
            yield f"{prefix}.{key}" , value.__func__
        elif isinstance(value , staticmethod):
            yield f"{prefix}.{key}" , value.__func__
        elif inspect.isfunction(value):
            yield f"{prefix}.{key}" , value


def iter_api_routines() -> Iterator[tuple[str , str , Callable[..., Any]]]:
    """Yield ``(module_name, qual_path, function)`` for public callables under ``src.api``."""
    import src.api as api_pkg

    for mod_info in pkgutil.walk_packages(api_pkg.__path__ , api_pkg.__name__ + "."):
        if mod_info.name.endswith("test_contract") or mod_info.name.endswith(".contract"):
            continue
        if mod_info.name.endswith(".interaction_runner"):
            continue
        try:
            module = importlib.import_module(mod_info.name)
        except Exception:
            continue
        mod_name = module.__name__
        for name , obj in module.__dict__.items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(obj) and getattr(obj , "__module__" , None) == mod_name:
                yield mod_name , name , obj
            elif inspect.isclass(obj) and getattr(obj , "__module__" , None) == mod_name:
                for path , fn in _walk_class_attrs(obj , name):
                    yield mod_name , path , fn


@dataclass(frozen = True)
class APIEndpoint:
    """Resolved API endpoint metadata for one callable."""
    module : str
    qual_path : str
    func : Callable[..., Any]
    interaction : dict[str, Any]

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.qual_path}"

    @property
    def task_id(self) -> str:
        return f"{self.qualname}@{int(datetime.now().timestamp())}"

    def execute(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def execute_with_script_tool(self, **kwargs: Any) -> Any:
        from src.proj.util.script.script_tool import ScriptTool
        tool = ScriptTool(task_name = self.qualname, source_mode = 'api', interaction = self.interaction)
        return tool(self.func)(**kwargs)

def iter_endpoints_with_interaction() -> Iterator[APIEndpoint]:
    """Yield callables whose docstring defines a non-empty ``[API Interaction]`` block."""
    for mod_name , path , fn in iter_api_routines():
        block = interaction_for_callable(fn)
        if block:
            yield APIEndpoint(module = mod_name , qual_path = path , func = fn , interaction = block)


def validate_all_api_interactions(*, strict_unknown_keys : bool = True) -> list[tuple[str , list[str]]]:
    """Validate every discovered interaction block; return list of ``(qual_path, errors)``."""
    bad : list[tuple[str , list[str]]] = []
    for rec in iter_endpoints_with_interaction():
        key = f"{rec.module}:{rec.qual_path}"
        errs = validate_interaction_schema(rec.interaction , strict_unknown_keys = strict_unknown_keys)
        if errs:
            bad.append((key , errs))
    return bad


def assert_all_api_contracts_ok(*, strict_unknown_keys : bool = True) -> None:
    """Raise AssertionError with details if any contract is invalid."""
    bad = validate_all_api_interactions(strict_unknown_keys = strict_unknown_keys)
    if bad:
        lines = [f"{k}: " + "; ".join(e) for k , e in bad]
        raise AssertionError("API [API Interaction] validation failed:\n" + "\n".join(lines))


def human_description_before_interaction(doc : str | None) -> str:
    """Return the human docstring slice before ``[API Interaction]:``."""
    if not doc:
        return ''
    if INTERACTION_HEADER not in doc:
        return doc.strip()
    return doc.split(INTERACTION_HEADER , 1)[0].strip()


def _annotation_repr(ann : Any) -> str | None:
    if ann is None or ann is inspect.Parameter.empty:
        return None
    if isinstance(ann , type):
        return ann.__name__
    return str(ann)


def describe_api_callable(obj : Any) -> dict[str, Any]:
    """
    Build description text, validated ``[API Interaction]`` schema, and parameter rows.

    Parameter rows merge ``override_arg_attr`` entries for the same name. ``cls`` / ``self``
    are omitted when they appear as the first explicit positional parameter (raw classmethod).
    """
    doc = inspect.getdoc(obj)
    schema = interaction_for_callable(obj)
    if schema is None:
        raise ValueError(f'no [API Interaction] block on {obj!r}')
    errs = validate_interaction_schema(schema)
    if errs:
        raise ValueError('; '.join(errs))
    sig = inspect.signature(obj)
    explicit = explicit_signature_parameters(sig)
    oaa : dict[str, Any] = schema.get('override_arg_attr') or {}
    items = list(explicit.items())
    if items and items[0][0] in ('cls' , 'self'):
        items = items[1:]
    params : list[dict[str, Any]] = []
    for name , p in items:
        default = None if p.default is inspect.Parameter.empty else p.default
        row : dict[str, Any] = {
            'name': name ,
            'annotation': _annotation_repr(p.annotation) ,
            'default': default ,
            'override': oaa.get(name) ,
        }
        params.append(row)
    return {
        'description': human_description_before_interaction(doc) ,
        'schema': schema ,
        'parameters': params ,
    }


__all__ = [
    "INTERACTION_HEADER" ,
    "ROLES" , "RISKS" , "PLATFORMS" , "EXECUTION_TIMES" , "MEMORY_USAGE" ,
    "parse_interaction_block" ,
    "validate_interaction_schema" ,
    "interaction_for_callable" ,
    "explicit_signature_parameters" ,
    "filter_kwargs_explicit_only" ,
    "bind_explicit_only" ,
    "iter_api_routines" ,
    "iter_endpoints_with_interaction" ,
    "validate_all_api_interactions" ,
    "assert_all_api_contracts_ok" ,
    "APIEndpoint" ,
    "human_description_before_interaction" ,
    "describe_api_callable" ,
]
