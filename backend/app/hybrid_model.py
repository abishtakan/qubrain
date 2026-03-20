from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn


DEFAULT_N_QUBITS = 6
DEFAULT_N_LAYERS = 2
DEFAULT_HIDDEN_DIM = 32
DEFAULT_HEAD_DIM = 16
DEFAULT_DROPOUT = 0.2
DEFAULT_TEMPERATURE = 0.5


def _build_quantum_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch")
    def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(index)) for index in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridQuantumClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout = dropout
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),
        )
        self.q_layer = _build_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def get_init_params(self) -> dict[str, int | float]:
        return {
            "n_features": self.n_features,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "temperature": self.temperature,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        quantum_input = (encoded + 1) * (math.pi / 2)
        quantum_output = self.q_layer(quantum_input)
        logits = self.head(quantum_output)
        return torch.sigmoid(logits / self.temperature).squeeze(-1)
