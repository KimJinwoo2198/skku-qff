from __future__ import annotations

import pennylane as qml
import pennylane.numpy as qnp

from pqc.model import TwoQubitPQC

from .config import EntangledPQCConfig


class EntangledTwoQubitPQC(TwoQubitPQC):
    """얽힘 게이트를 포함하는 2-큐비트 PQC."""

    config: EntangledPQCConfig

    def __init__(self, config: EntangledPQCConfig):
        self.config = config
        super().__init__(config)

    @staticmethod
    def _single_qubit_layer(inputs: qnp.ndarray, block_params: qnp.ndarray) -> None:
        bit_inputs = [int(bit) for bit in inputs]
        for wire in range(2):
            theta, phi, lam, alpha, beta = block_params[wire]
            if bit_inputs[0]:
                qml.RY(alpha, wires=wire)
            if len(bit_inputs) > 1 and bit_inputs[1]:
                qml.RY(beta, wires=wire)
            qml.RY(theta, wires=wire)
            qml.RZ(phi, wires=wire)
            qml.RY(lam, wires=wire)

    def _apply_entangler(self) -> None:
        ctrl, tgt = self.config.entangle_order
        self._run_entangler(ctrl, tgt)
        if self.config.mirror_entangler:
            self._run_entangler(tgt, ctrl)

    def _run_entangler(self, control: int, target: int) -> None:
        gate = self.config.entangler.upper()
        wires = (control, target)
        if gate == "CNOT":
            qml.CNOT(wires=wires)
        elif gate == "CZ":
            qml.CZ(wires=wires)
        elif gate == "ISWAP":
            qml.ISWAP(wires=wires)
        elif gate == "SWAP":
            qml.SWAP(wires=wires)
        else:  # pragma: no cover - 설정 오류
            raise ValueError(f"지원하지 않는 얽힘 게이트: {self.config.entangler}")

    def _circuit(self, inputs: qnp.ndarray, params: qnp.ndarray) -> qnp.ndarray:
        self._basis_encoding(inputs)
        for block in range(params.shape[0]):
            block_params = params[block]
            self._single_qubit_layer(inputs, block_params)
            self._apply_entangler()
        return qml.expval(qml.PauliZ(0))

