from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from pqc.model import PQCConfig


@dataclass(frozen=True)
class EntangledPQCConfig(PQCConfig):
    """얽힘 게이트를 포함한 2-큐비트 PQC 설정."""

    entangler: str = "CNOT"
    entangle_order: Tuple[int, int] = (0, 1)
    mirror_entangler: bool = True

    def __post_init__(self) -> None:
        if len(self.entangle_order) != 2:
            raise ValueError("entangle_order는 (control, target) 두 요소여야 합니다.")
        ctrl, tgt = self.entangle_order
        if ctrl == tgt:
            raise ValueError("control과 target은 서로 달라야 합니다.")
        if ctrl not in {0, 1} or tgt not in {0, 1}:
            raise ValueError("entangle_order는 0과 1로만 구성되어야 합니다.")

