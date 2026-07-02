# -*- coding: utf-8 -*-
import os
import yaml

class _CfgNode:
    def __init__(self, data: dict):
        for key, value in data.items():
            attr_name = str(key)
            if isinstance(value, dict):
                if all(isinstance(k, str) for k in value.keys()):
                    setattr(self, attr_name, _CfgNode(value))
                else:
                    setattr(self, attr_name, value)
            else:
                setattr(self, attr_name, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"CfgNode({items})"

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, _CfgNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)

def _find_config() -> str:
    # 1. Check around execution scripts or main
    import __main__
    if hasattr(__main__, "__file__"):
        main_dir = os.path.dirname(os.path.abspath(__main__.__file__))
        candidate = os.path.join(main_dir, "config.yaml")
        if os.path.exists(candidate):
            return candidate
        parent_candidate = os.path.join(os.path.dirname(main_dir), "config.yaml")
        if os.path.exists(parent_candidate):
            return parent_candidate

    # 2. Check current working directory
    cwd = os.getcwd()
    candidate = os.path.join(cwd, "config.yaml")
    if os.path.exists(candidate):
        return candidate

    # 3. Walk up from CWD (max 5 levels)
    d = cwd
    for _ in range(5):
        d = os.path.dirname(d)
        candidate = os.path.join(d, "config.yaml")
        if os.path.exists(candidate):
            return candidate

    # 4. Try system/repo root fallback
    here = os.path.dirname(os.path.abspath(__file__))
    # core/utils/config_loader.py -> walk up 3 levels to find root config
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    candidate = os.path.join(repo_root, "config.yaml")
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError("config.yaml not found in CWD, script directory, or parent paths.")

def load_config(path: str = None) -> _CfgNode:
    if path is None:
        path = _find_config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _CfgNode(raw)

cfg = load_config()
