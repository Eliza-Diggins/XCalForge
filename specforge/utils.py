"""
Utility functions for SpecForge.
"""
from typing import Union
import importlib
import logging
import sys
import yaml
from pathlib import Path
from collections.abc import MutableMapping

class ConfigManager(MutableMapping):
    def __init__(self, path: Union[str, Path], autosave: bool = True):
        self._path = Path(path).expanduser().resolve()
        self._autosave = autosave
        self._data = self._load()

    def _load(self):
        if not self._path.exists():
            return {}
        with open(self._path, "r") as f:
            return yaml.safe_load(f) or {}

    def _save(self):
        with open(self._path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False)

    def _traverse(self, key: str, create_missing=False):
        keys = key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                if create_missing:
                    node[k] = {}
                else:
                    raise KeyError(f"'{k}' not found in config.")
            node = node[k]
        return node, keys[-1]

    def __getitem__(self, key):
        node, final_key = self._traverse(key)
        return node[final_key]

    def __setitem__(self, key, value):
        node, final_key = self._traverse(key, create_missing=True)
        node[final_key] = value
        if self._autosave:
            self._save()

    def __delitem__(self, key):
        node, final_key = self._traverse(key)
        del node[final_key]
        if self._autosave:
            self._save()

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"<ConfigManager path={self._path} data={self._data}>"

    def to_dict(self):
        return self._data

def get_xspec():
    """
    Safely load the XSPEC PyXspec module, avoiding re-imports that can cause CLI-level deadlocks.

    Returns
    -------
    module
        The imported `xspec` module.

    Raises
    ------
    ImportError
        If the `xspec` module is not available or cannot be loaded.

    Notes
    -----
    - PyXspec can enter an unstable state if `xspec` is imported more than once across
      subprocesses or reloaded dynamically (e.g., during hot-reload or interactive development).
    - This function ensures that the module is loaded only once and reuses it afterward.
    - It is *strongly recommended* that all modules use `get_xspec()` rather than
      importing `xspec` directly.

    Example
    -------
    >>> xspec = get_xspec()
    >>> xspec.AllData.clear()
    """
    if 'xspec' in sys.modules:
        return sys.modules['xspec']
    
    try:
        xspec = importlib.import_module('xspec')
        return xspec
    except ImportError as e:
        logging.error("Failed to import xspec module. Ensure PyXspec is installed and configured.")
        raise e