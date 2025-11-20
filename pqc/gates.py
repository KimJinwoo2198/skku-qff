from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Callable

import pennylane.numpy as qnp


class LogicGate(Enum):
    """2입력 1출력 고전 논리 게이트 정의."""

    AND = "AND"
    OR = "OR"
    NAND = "NAND"
    NOR = "NOR"
    XOR = "XOR"
    XNOR = "XNOR"


GateFunction = Callable[[int, int], int]


GATE_FUNCTIONS: dict[LogicGate, GateFunction] = {
    LogicGate.AND: lambda a, b: a & b,
    LogicGate.OR: lambda a, b: a | b,
    LogicGate.NAND: lambda a, b: int(not (a & b)),
    LogicGate.NOR: lambda a, b: int(not (a | b)),
    LogicGate.XOR: lambda a, b: a ^ b,
    LogicGate.XNOR: lambda a, b: int(not (a ^ b)),
}


def truth_table_inputs(num_qubits: int = 2) -> list[tuple[int, ...]]:
    """2진 입력 조합을 반환한다."""

    return list(product((0, 1), repeat=num_qubits))


def build_dataset(gate: LogicGate) -> list[tuple[qnp.ndarray, float]]:
    """해당 게이트의 진리표를 양자 회로 학습용 데이터로 변환."""

    dataset: list[tuple[qnp.ndarray, float]] = []
    fn = GATE_FUNCTIONS[gate]
    for bits in truth_table_inputs():
        dataset.append((qnp.array(bits, dtype=float), float(fn(*bits))))
    return dataset

