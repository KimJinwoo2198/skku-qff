"""각도 인코딩 기반 2-큐비트 PQC 패키지."""

from importlib import import_module
from typing import Any

_EXPORT_MAP = {
    "AnglePQCConfig": ("pqc.angle.config", "AnglePQCConfig"),
    "AngleEncodedTwoQubitPQC": ("pqc.angle.model", "AngleEncodedTwoQubitPQC"),
    "AngleTrainingResult": ("pqc.angle.workflow", "AngleTrainingResult"),
    "train_angle_gate": ("pqc.angle.workflow", "train_angle_gate"),
    "log_angle_result": ("pqc.angle.workflow", "log_angle_result"),
    "run_angle_experiments": ("pqc.angle.workflow", "run_angle_experiments"),
}

__all__ = tuple(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'pqc.angle' has no attribute '{name}'") from None
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

