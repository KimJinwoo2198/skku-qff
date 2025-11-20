"""얽힘 기반 2-큐비트 PQC 패키지."""

from importlib import import_module
from typing import Any


_EXPORT_MAP = {
    "EntangledTwoQubitPQC": ("pqc.tangle.model", "EntangledTwoQubitPQC"),
    "EntangledTrainingResult": ("pqc.tangle.workflow", "EntangledTrainingResult"),
    "train_entangled_gate": ("pqc.tangle.workflow", "train_entangled_gate"),
    "log_entangled_result": ("pqc.tangle.workflow", "log_entangled_result"),
    "run_entangled_experiments": ("pqc.tangle.workflow", "run_entangled_experiments"),
}

__all__ = tuple(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'pqc.tangle' has no attribute '{name}'") from None
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

