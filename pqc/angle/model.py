from __future__ import annotations

import math

import pennylane as qml
import pennylane.numpy as qnp

from pqc.model import TwoQubitPQC

from .config import AnglePQCConfig


class AngleEncodedTwoQubitPQC(TwoQubitPQC):
    """각도 인코딩을 사용하는 2-큐비트 PQC."""

    config: AnglePQCConfig

    def __init__(self, config: AnglePQCConfig):
        self.config = config
        super().__init__(config)

    def _angle_encoding(self, inputs: qnp.ndarray) -> None:
        axis = self.config.angle_axis.upper()
        for wire, bit in enumerate(inputs):
            angle = self.config.angle_scale * float(bit) + self.config.angle_bias
            if axis == "RX":
                qml.RX(angle, wires=wire)
            elif axis == "RY":
                qml.RY(angle, wires=wire)
            elif axis == "RZ":
                qml.RZ(angle, wires=wire)

    @staticmethod
    def _ansatz_block(params: qnp.ndarray) -> None:
        for wire in range(2):
            theta, phi, lam, alpha, beta = params[wire]
            qml.RY(theta, wires=wire)
            qml.RZ(phi, wires=wire)
            qml.RY(lam, wires=wire)
            qml.RX(alpha, wires=wire)
            qml.RZ(beta, wires=wire)
        qml.CRX(math.pi / 2, wires=(0, 1))
        qml.CRX(math.pi / 2, wires=(1, 0))

    def _circuit(self, inputs: qnp.ndarray, params: qnp.ndarray) -> qnp.ndarray:
        self._angle_encoding(inputs)
        for block in range(params.shape[0]):
            self._ansatz_block(params[block])
        return qml.expval(qml.PauliZ(0))

