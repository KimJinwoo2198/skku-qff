from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
import pennylane.numpy as qnp


@dataclass(frozen=True)
class PQCConfig:
    """PQC 학습 설정."""

    learning_rate: float = 0.2
    max_steps: int = 400
    seed: int = 7
    shots: int | None = None
    convergence_tol: float = 1e-3
    num_blocks: int = 2


class TwoQubitPQC:
    """얽힘 없이 2-큐비트 PQC를 구성해 고전 게이트를 모방."""

    def __init__(self, config: PQCConfig):
        self.config = config
        self.dev = qml.device("default.qubit", wires=2, shots=config.shots)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        self.params = self._init_params(config.seed)

    def _init_params(self, seed: int) -> qnp.ndarray:
        rng = np.random.default_rng(seed)
        shape = (self.config.num_blocks, 2, 3)
        return qnp.array(rng.uniform(-np.pi, np.pi, size=shape), requires_grad=True)

    @staticmethod
    def _basis_encoding(bits: Iterable[float]) -> None:
        for wire, bit in enumerate(bits):
            if int(bit) == 1:
                qml.PauliX(wires=wire)

    @staticmethod
    def _ansatz_layer(params: qnp.ndarray) -> None:
        for block in range(params.shape[0]):
            for wire in range(2):
                theta, phi, lam = params[block, wire]
                qml.RY(theta, wires=wire)
                qml.RZ(phi, wires=wire)
                qml.RY(lam, wires=wire)

    def _circuit(self, inputs: Sequence[float], params: qnp.ndarray) -> qnp.ndarray:
        self._basis_encoding(inputs)
        self._ansatz_layer(params)
        return qml.expval(qml.PauliZ(0))

    @staticmethod
    def _expval_to_prob(expval: qnp.ndarray) -> qnp.ndarray:
        return 0.5 * (1 - expval)

    def loss(self, dataset: list[tuple[qnp.ndarray, float]], params: qnp.ndarray) -> qnp.ndarray:
        errors: list[qnp.ndarray] = []
        for features, target in dataset:
            expval = self.qnode(features, params)
            prob = self._expval_to_prob(expval)
            errors.append((prob - target) ** 2)
        return qnp.stack(errors).mean()

    def fit(self, dataset: list[tuple[qnp.ndarray, float]]) -> list[float]:
        optimizer = qml.AdamOptimizer(stepsize=self.config.learning_rate)
        params = self.params
        history: list[float] = []

        for _ in range(self.config.max_steps):
            params, loss_val = optimizer.step_and_cost(lambda p: self.loss(dataset, p), params)
            history.append(float(loss_val))
            if loss_val < self.config.convergence_tol:
                break

        self.params = params
        return history

    def predict_probability(self, inputs: qnp.ndarray) -> float:
        expval = self.qnode(inputs, self.params)
        prob = self._expval_to_prob(expval)
        return float(prob)

    def evaluate(
        self, dataset: list[tuple[qnp.ndarray, float]]
    ) -> tuple[list[float], list[int], list[int]]:
        probabilities: list[float] = []
        predictions: list[int] = []
        targets: list[int] = []

        for features, target in dataset:
            prob = self.predict_probability(features)
            probabilities.append(prob)
            predictions.append(int(prob >= 0.5))
            targets.append(int(target))

        return probabilities, predictions, targets

