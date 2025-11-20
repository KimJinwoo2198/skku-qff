from __future__ import annotations

from typing import Sequence, Tuple, TYPE_CHECKING

import numpy as np

from qiskit import QuantumCircuit

if TYPE_CHECKING:  # 순환 참조 회피용
    from pqc.workflow import TrainingResult


def _basis_encode(circuit: QuantumCircuit, inputs: Sequence[int]) -> None:
    for wire, bit in enumerate(inputs):
        if int(bit):
            circuit.x(wire)


def _ansatz_layers(circuit: QuantumCircuit, params: np.ndarray, inputs: Sequence[int]) -> None:
    encoded_bits = tuple(int(bit) for bit in inputs)
    for block in range(params.shape[0]):
        for wire in range(2):
            theta, phi, lam, alpha, beta = params[block, wire]
            if encoded_bits[0]:
                circuit.ry(alpha, wire)
            if len(encoded_bits) > 1 and encoded_bits[1]:
                circuit.ry(beta, wire)
            circuit.ry(theta, wire)
            circuit.rz(phi, wire)
            circuit.ry(lam, wire)


def build_quantum_circuit(
    params: np.ndarray,
    inputs: Sequence[int],
    circuit_name: str = "TwoQubitPQC",
) -> QuantumCircuit:
    circuit = QuantumCircuit(2, 1, name=circuit_name)
    _basis_encode(circuit, inputs)
    _ansatz_layers(circuit, params, inputs)
    circuit.barrier()
    circuit.measure(0, 0)
    return circuit


def display_qiskit_report(
    result: "TrainingResult",
    inputs: Tuple[int, int] = (1, 1),
) -> None:
    params = np.asarray(result.params, dtype=float)
    circuit = build_quantum_circuit(params, inputs, circuit_name=f"{result.gate.value}_PQC")
    diagram = circuit.draw(output="text", fold=120)
    print(f"\n[{result.gate.value}] Qiskit 회로 시각화 (입력 {inputs})")
    print(diagram)

