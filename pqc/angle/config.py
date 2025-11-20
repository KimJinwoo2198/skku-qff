from __future__ import annotations

from dataclasses import dataclass

from pqc.model import PQCConfig


@dataclass(frozen=True)
class AnglePQCConfig(PQCConfig):
    """각도 인코딩 기반 PQC 설정."""

    angle_axis: str = "RY"
    angle_scale: float = 3.141592653589793
    angle_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.angle_axis.upper() not in {"RX", "RY", "RZ"}:
            raise ValueError("angle_axis는 RX/RY/RZ 중 하나여야 합니다.")

