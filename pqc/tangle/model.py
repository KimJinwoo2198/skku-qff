from __future__ import annotations

import pennylane as qml
import pennylane.numpy as qnp

from pqc.model import PQCConfig, TwoQubitPQC


class EntangledTwoQubitPQC(TwoQubitPQC):
    """얽힘과 비얽힘 게이트를 하나의 앤사츠 레이어에서 통합한 2-큐비트 PQC."""

    def __init__(self, config: PQCConfig | None = None):
        """필수 하이퍼파라미터가 없다면 기본 PQC 설정을 사용."""
        super().__init__(config or PQCConfig())

    @staticmethod
    def _basis_encoding(bits: qnp.ndarray) -> None:
        """Figure 1의 첫 단계: 입력 비트를 Pauli-X로 인코딩."""
        for wire, bit in enumerate(bits):
            if int(bit) == 1:
                qml.PauliX(wires=wire)

    @staticmethod
    def _ansatz_layer(params: qnp.ndarray) -> None:
        """단일 레이어에서 회전 게이트와 얽힘 게이트를 모두 적용."""
        for block in range(params.shape[0]):
            block_params = params[block]

            # 1) 비얽힘(단일 큐비트) 회전
            for wire in range(2):
                theta, phi, lam = block_params[wire]
                qml.RY(theta, wires=wire)
                qml.RZ(phi, wires=wire)
                qml.RY(lam, wires=wire)

            # 2) 얽힘 게이트 체인 (파라미터 재활용)
            zz_angle = 0.5 * (block_params[0, 0] + block_params[1, 0])
            xx_angle = 0.5 * (block_params[0, 2] + block_params[1, 2])
            qml.IsingZZ(zz_angle, wires=(0, 1))
            qml.CNOT(wires=(0, 1))
            qml.IsingXX(xx_angle, wires=(0, 1))
            qml.CNOT(wires=(1, 0))

    def _circuit(self, inputs: qnp.ndarray, params: qnp.ndarray) -> qnp.ndarray:
        """Figure 1 구조(기저 인코딩 → 앤사츠 → 측정)를 명시적으로 구현."""
        self._basis_encoding(inputs)
        self._ansatz_layer(params)
        return qml.expval(qml.PauliZ(0))