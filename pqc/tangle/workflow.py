from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane.numpy as qnp

from pqc.gates import LogicGate, build_dataset, truth_table_inputs
from pqc.model import PQCConfig
from pqc.report import display_qiskit_report

from .model import EntangledTwoQubitPQC


@dataclass
class EntangledTrainingResult:
    gate: LogicGate
    final_loss: float
    accuracy: float
    converged: bool
    params: qnp.ndarray
    probabilities: list[float]
    predictions: list[int]
    targets: list[int]
    loss_history: list[float]


def train_entangled_gate(gate: LogicGate, config: PQCConfig | None = None) -> EntangledTrainingResult:
    dataset = build_dataset(gate)
    pqc = EntangledTwoQubitPQC(config)
    history = pqc.fit(dataset)
    probabilities, predictions, targets = pqc.evaluate(dataset)
    accuracy = sum(int(p == t) for p, t in zip(predictions, targets)) / len(targets)
    final_loss = history[-1] if history else float("inf")
    tol = pqc.config.convergence_tol
    converged = final_loss < tol
    return EntangledTrainingResult(
        gate=gate,
        final_loss=final_loss,
        accuracy=accuracy,
        converged=converged,
        params=pqc.params,
        probabilities=probabilities,
        predictions=predictions,
        targets=targets,
        loss_history=history,
    )


def log_entangled_result(result: EntangledTrainingResult, config: PQCConfig) -> None:
    status = "성공" if result.accuracy == 1.0 else "제한"
    print(f"\n[Entangled {result.gate.value}] 학습 {status}")
    print(f"  최종 손실: {result.final_loss:.6f} (수렴 기준 {config.convergence_tol})")
    print(f"  정확도: {result.accuracy * 100:.1f}%")
    print(f"  블록 수: {config.num_blocks} / 러닝레이트: {config.learning_rate}")
    print(f"  매개변수: {np.round(result.params, 3)}")
    print("  입력별 추정 확률/예측/정답:")
    for bits, prob, pred, target in zip(truth_table_inputs(), result.probabilities, result.predictions, result.targets):
        print(f"    입력 {bits} -> P(1)={prob:.3f} / 예측={pred} / 정답={target}")


def run_entangled_experiments(config: PQCConfig | None = None) -> None:
    ent_config = config or PQCConfig(
        learning_rate=0.1,
        max_steps=800,
        num_blocks=4,
        seed=13,
    )
    gates_to_learn = [
        LogicGate.AND,
        LogicGate.OR,
        LogicGate.NAND,
        LogicGate.NOR,
        LogicGate.XOR,
        LogicGate.XNOR,
    ]
    for gate in gates_to_learn:
        result = train_entangled_gate(gate, ent_config)
        log_entangled_result(result, ent_config)
        display_qiskit_report(result)

