"""
Deep Project 1: Classical vs Quantum Radical Pair — Unified Comparison Engine
=============================================================================
Solves the same physical problem through two radically different methods:

  1. EXACT (classical): full diagonalization / matrix exponential
  2. TROTTER (quantum): Trotterized unitary on quantum circuit

Then analyzes:
  - Trotterization error ∥U_exact − U_trotter∥ vs n_steps
  - Shot noise scaling (1024 .. 65536 shots)
  - NISQ noise threshold (how many Trotter steps survive)
  - Resource estimation (#gates, #qubits) → classical/quantum crossover

All in natural dimensionless units where HFC_A = 1 (matching arXiv:2406.12986).
"""

import numpy as np
from scipy.linalg import expm, norm
from typing import Dict, Tuple, List, Optional

# ═══════════════════════════════════════════════════════════════
#  Pauli / spin-1/2 infrastructure
# ═══════════════════════════════════════════════════════════════
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SP = np.array([[0, 1], [0, 0]], dtype=complex)
SM = np.array([[0, 0], [1, 0]], dtype=complex)


def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def embed_op(op, target, total):
    ops = [I2] * total
    ops[target] = op
    return tensor(*ops)


def embed_2q(A, i, B, j, total):
    ops = [I2] * total
    ops[i], ops[j] = A, B
    return tensor(*ops)


# ═══════════════════════════════════════════════════════════════
#  System specification
# ═══════════════════════════════════════════════════════════════

class RPSystem:
    """
    Radical pair system specification.

    In natural units (HFC_A = 1), the physical Hamiltonian is:
        H = ω·(Sz_A + Sz_B) + J·S_A·S_B + Σ A_k·S_e·I_k

    Parameters (all dimensionless):
        omega  : Zeeman strength (typically 0.01–1.0)
        J      : Exchange coupling
        hfc_A  : list of HFC strengths on radical A
        hfc_B  : list of HFC strengths on radical B
        k_S, k_T : recombination rates (for yield calculation)
    """
    def __init__(self, omega=0.1, J=0.01,
                 hfc_A=None, hfc_B=None,
                 k_S=0.5, k_T=0.5):
        self.omega = omega
        self.J = J
        self.hfc_A = list(hfc_A) if hfc_A else []
        self.hfc_B = list(hfc_B) if hfc_B else []
        self.k_S = k_S
        self.k_T = k_T
        self.nA = len(self.hfc_A)
        self.nB = len(self.hfc_B)
        self.n_qubits = 2 + self.nA + self.nB
        self.dim = 2 ** self.n_qubits

    def build_H(self) -> np.ndarray:
        """Full Hamiltonian matrix (dim × dim)."""
        n, d = self.n_qubits, self.dim
        H = np.zeros((d, d), dtype=complex)

        # Zeeman
        ω = self.omega
        H += ω * embed_op(SZ, 0, n) / 2
        H += ω * embed_op(SZ, 1, n) / 2

        # Exchange
        J = self.J
        H += J * embed_2q(SP, 0, SM, 1, n) / 2
        H += J * embed_2q(SM, 0, SP, 1, n) / 2
        H += J * embed_2q(SZ, 0, SZ, 1, n) / 4

        # Hyperfine
        for k, a in enumerate(self.hfc_A):
            q = 2 + k
            H += a * embed_2q(SP, 0, SM, q, n) / 2
            H += a * embed_2q(SM, 0, SP, q, n) / 2
            H += a * embed_2q(SZ, 0, SZ, q, n) / 4
        for k, a in enumerate(self.hfc_B):
            q = 2 + self.nA + k
            H += a * embed_2q(SP, 1, SM, q, n) / 2
            H += a * embed_2q(SM, 1, SP, q, n) / 2
            H += a * embed_2q(SZ, 1, SZ, q, n) / 4

        return H

    def singlet_vec(self) -> np.ndarray:
        """|S⟩ = (|01⟩ − |10⟩)/√2 ⊗ |00…0⟩."""
        v = np.zeros(self.dim, dtype=complex)
        v[2 ** (self.n_qubits - 2)] = 1 / np.sqrt(2)   # |01⟩
        v[2 ** (self.n_qubits - 1)] = -1 / np.sqrt(2)  # |10⟩
        return v

    def singlet_proj(self) -> np.ndarray:
        v = self.singlet_vec()
        return np.outer(v, v.conj())

    def __repr__(self):
        return (f"RPSystem(ω={self.omega}, J={self.J}, "
                f"HFC_A={self.hfc_A}, HFC_B={self.hfc_B}, "
                f"n_qubits={self.n_qubits})")


# ═══════════════════════════════════════════════════════════════
#  1. EXACT (classical) evolution
# ═══════════════════════════════════════════════════════════════

def exact_evolution(sys: RPSystem, t_values: np.ndarray) -> np.ndarray:
    """
    Exact singlet probability P_S(t) via full matrix exponentiation.
    Gold standard — limited only by floating-point precision.

    Returns P_S(t) for each t in t_values.
    """
    H = sys.build_H()
    psi0 = sys.singlet_vec()
    P_op = sys.singlet_proj()
    result = np.zeros(len(t_values), dtype=float)
    for i, t in enumerate(t_values):
        U = expm(-1j * H * t)
        psi = U @ psi0
        result[i] = np.real(np.conj(psi) @ P_op @ psi)
    return result


# ═══════════════════════════════════════════════════════════════
#  2. TROTTER (quantum circuit) evolution — STATEVECTOR
# ═══════════════════════════════════════════════════════════════

def trotter_evolution(sys: RPSystem, t_values: np.ndarray,
                       n_trotter: int = 15) -> np.ndarray:
    """
    Trotterized evolution (statevector, no shot noise).

    U(t) ≈ [exp(-i·H_HFC·dt) · exp(-i·H_Ex·dt) · exp(-i·H_Z·dt)]^{n_trotter}
    """
    n, d = sys.n_qubits, sys.dim

    H_Z = sys.omega * (embed_op(SZ, 0, n) + embed_op(SZ, 1, n)) / 2
    H_Ex = np.zeros((d, d), dtype=complex)
    H_Ex += sys.J * embed_2q(SP, 0, SM, 1, n) / 2
    H_Ex += sys.J * embed_2q(SM, 0, SP, 1, n) / 2
    H_Ex += sys.J * embed_2q(SZ, 0, SZ, 1, n) / 4
    H_HFC = np.zeros((d, d), dtype=complex)
    for k, a in enumerate(sys.hfc_A):
        q = 2 + k
        H_HFC += a * embed_2q(SP, 0, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 0, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 0, SZ, q, n) / 4
    for k, a in enumerate(sys.hfc_B):
        q = 2 + sys.nA + k
        H_HFC += a * embed_2q(SP, 1, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 1, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 1, SZ, q, n) / 4

    psi0 = sys.singlet_vec()
    P_op = sys.singlet_proj()
    result = np.zeros(len(t_values), dtype=float)

    for i, t in enumerate(t_values):
        dt = t / n_trotter
        U_Z = expm(-1j * H_Z * dt)
        U_Ex = expm(-1j * H_Ex * dt)
        U_HFC = expm(-1j * H_HFC * dt)

        psi = psi0.copy()
        for _ in range(n_trotter):
            psi = U_HFC @ U_Ex @ U_Z @ psi
        result[i] = np.real(np.conj(psi) @ P_op @ psi)

    return result


# ═══════════════════════════════════════════════════════════════
#  3. NISQ (noisy) evolution — AERSIMULATOR with SHOTS
# ═══════════════════════════════════════════════════════════════

def nisq_evolution(sys: RPSystem, t_values: np.ndarray,
                    n_trotter: int = 15, shots: int = 8192,
                    p1: float = 0.0, p2: float = 0.0) -> np.ndarray:
    """
    Noisy quantum circuit simulation with shot noise.

    Parameters
    ----------
    p1, p2 : depolarizing error probability for 1Q / 2Q gates
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.primitives.containers.bindings_array import BindingsArray
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    n, d = sys.n_qubits, sys.dim
    H_Z = sys.omega * (embed_op(SZ, 0, n) + embed_op(SZ, 1, n)) / 2
    H_Ex = np.zeros((d, d), dtype=complex)
    H_Ex += sys.J * embed_2q(SP, 0, SM, 1, n) / 2
    H_Ex += sys.J * embed_2q(SM, 0, SP, 1, n) / 2
    H_Ex += sys.J * embed_2q(SZ, 0, SZ, 1, n) / 4
    H_HFC = np.zeros((d, d), dtype=complex)
    for k, a in enumerate(sys.hfc_A):
        q = 2 + k
        H_HFC += a * embed_2q(SP, 0, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 0, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 0, SZ, q, n) / 4
    for k, a in enumerate(sys.hfc_B):
        q = 2 + sys.nA + k
        H_HFC += a * embed_2q(SP, 1, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 1, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 1, SZ, q, n) / 4

    singlet_gate = _singlet_prep_circuit(n)

    # Noise model
    noise_model = None
    if p1 > 0 or p2 > 0:
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1), ['x', 'h', 'z', 'rx', 'ry', 'rz', 'sx'])
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), ['cx'])

    sim = AerSimulator(noise_model=noise_model)
    all_zeros = '0' * n
    result = np.zeros(len(t_values), dtype=float)

    for i, t in enumerate(t_values):
        dt = t / n_trotter
        U_Z = expm(-1j * H_Z * dt)
        U_Ex = expm(-1j * H_Ex * dt)
        U_HFC = expm(-1j * H_HFC * dt)
        U_step = U_HFC @ U_Ex @ U_Z

        qc = QuantumCircuit(n)
        qc.compose(singlet_gate, inplace=True)
        step_gate = UnitaryGate(U_step, label='U_step')
        for _ in range(n_trotter):
            qc.append(step_gate, range(n))
        qc.compose(singlet_gate.inverse(), inplace=True)
        qc.measure_all()

        qc_t = transpile(qc, sim, optimization_level=1)
        counts = sim.run(qc_t, shots=shots).result().get_counts()
        result[i] = counts.get(all_zeros, 0) / shots

    return result


def _singlet_prep_circuit(n_qubits):
    """Circuit that prepares |S⟩ = (|01⟩−|10⟩)/√2 ⊗ |00…0⟩ from |0…0⟩."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    return qc


# ═══════════════════════════════════════════════════════════════
#  4. ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════

def trotter_error_analysis(sys: RPSystem, t_final: float = 50.0,
                            trotter_list: Optional[List[int]] = None) -> Dict:
    """
    Measure Trotter error vs exact for a range of Trotter step counts.

    Metrics:
      - mean_abs_error = mean|P_exact − P_trotter|
      - max_abs_error  = max|P_exact − P_trotter|
      - unitary_error   = ∥U_exact − U_trotter∥_F / ∥U_exact∥_F

    Returns dict with keys: n_steps, mean_error, max_error, unitary_error
    """
    if trotter_list is None:
        trotter_list = [1, 3, 5, 10, 15, 25, 40, 60, 100]
    t_values = np.linspace(0, t_final, 30)

    P_exact = exact_evolution(sys, t_values)
    H = sys.build_H()
    U_exact = expm(-1j * H * t_final)

    results = {}
    for n in trotter_list:
        P_trot = trotter_evolution(sys, t_values, n_trotter=n)
        U_trot = _trotter_unitary(sys, t_final, n)
        u_err = norm(U_exact - U_trot, 'fro') / norm(U_exact, 'fro')
        results[n] = {
            'mean_error': float(np.mean(np.abs(P_exact - P_trot))),
            'max_error': float(np.max(np.abs(P_exact - P_trot))),
            'unitary_error': float(u_err),
            'P_S': P_trot
        }

    # Also compute convergence rate (power law fit)
    ns = np.array(list(results.keys()))
    errs = np.array([r['mean_error'] for r in results.values()])
    # Fit: error ~ C · n^{-α}
    coeffs = np.polyfit(np.log(ns[errs > 0]), np.log(errs[errs > 0]), 1)
    alpha = -coeffs[0]  # convergence exponent

    return {
        'results': results,
        'P_exact': P_exact,
        't_values': t_values,
        'convergence_rate': float(alpha),
        'n_15_error': float(results[15]['mean_error']),
        'sufficient_steps': int(np.min([n for n in ns
                                         if results[n]['mean_error'] < 0.01]))
    }


def _trotter_unitary(sys: RPSystem, t: float, n_trotter: int) -> np.ndarray:
    """Compute the Trotterized unitary matrix."""
    dt = t / n_trotter
    n, d = sys.n_qubits, sys.dim
    H_Z = sys.omega * (embed_op(SZ, 0, n) + embed_op(SZ, 1, n)) / 2
    H_Ex = np.zeros((d, d), dtype=complex)
    H_Ex += sys.J * embed_2q(SP, 0, SM, 1, n) / 2
    H_Ex += sys.J * embed_2q(SM, 0, SP, 1, n) / 2
    H_Ex += sys.J * embed_2q(SZ, 0, SZ, 1, n) / 4
    H_HFC = np.zeros((d, d), dtype=complex)
    for k, a in enumerate(sys.hfc_A):
        q = 2 + k
        H_HFC += a * embed_2q(SP, 0, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 0, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 0, SZ, q, n) / 4
    for k, a in enumerate(sys.hfc_B):
        q = 2 + sys.nA + k
        H_HFC += a * embed_2q(SP, 1, SM, q, n) / 2
        H_HFC += a * embed_2q(SM, 1, SP, q, n) / 2
        H_HFC += a * embed_2q(SZ, 1, SZ, q, n) / 4
    U_Z = expm(-1j * H_Z * dt)
    U_Ex = expm(-1j * H_Ex * dt)
    U_HFC = expm(-1j * H_HFC * dt)
    U = np.eye(d, dtype=complex)
    for _ in range(n_trotter):
        U = U_HFC @ U_Ex @ U_Z @ U
    return U


# ═══════════════════════════════════════════════════════════════
#  5. SHOT NOISE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def shot_noise_analysis(sys: RPSystem, t_final: float = 50.0,
                          n_trotter: int = 15, n_time_points: int = 10,
                          shot_list: Optional[List[int]] = None) -> Dict:
    """
    Measure how shot count affects accuracy.

    Returns dict: {n_shots: {mean_error, std_error}}
    """
    if shot_list is None:
        shot_list = [1024, 2048, 4096, 8192, 16384, 65536, 131072]
    t_values = np.linspace(0, t_final, n_time_points)

    P_exact = exact_evolution(sys, t_values)
    results = {}
    for shots in shot_list:
        P_nisq = nisq_evolution(sys, t_values, n_trotter=n_trotter, shots=shots)
        results[shots] = {
            'mean_error': float(np.mean(np.abs(P_nisq - P_exact))),
            'P_S': P_nisq
        }

    return {'results': results, 'P_exact': P_exact, 't_values': t_values}


# ═══════════════════════════════════════════════════════════════
#  6. NISQ NOISE THRESHOLD
# ═══════════════════════════════════════════════════════════════

def nisq_noise_threshold(sys: RPSystem, t_final: float = 30.0,
                          trotter_list: Optional[List[int]] = None,
                          noise_levels: Optional[List[Tuple]] = None) -> Dict:
    """
    For each noise level, find how many Trotter steps are feasible
    before noise destroys the signal.

    Returns: {noise_label: {n_steps: mean_error}}
    """
    if trotter_list is None:
        trotter_list = [1, 3, 5, 10, 15, 25]
    if noise_levels is None:
        noise_levels = [
            ('ideal', 0.0, 0.0),
            ('low (p₁=1e-3, p₂=1e-2)', 1e-3, 1e-2),
            ('mid (p₁=5e-3, p₂=2.5e-2)', 5e-3, 2.5e-2),
            ('high (p₁=1e-2, p₂=5e-2)', 1e-2, 5e-2),
        ]

    t_values = np.linspace(0, t_final, 10)
    P_exact = exact_evolution(sys, t_values)
    all_results = {}

    for label, p1, p2 in noise_levels:
        noise_results = {}
        for n in trotter_list:
            P_noisy = nisq_evolution(sys, t_values, n_trotter=n,
                                      shots=8192, p1=p1, p2=p2)
            err = float(np.mean(np.abs(P_noisy - P_exact)))
            noise_results[n] = err
        all_results[label] = noise_results

    return {'results': all_results, 'P_exact': P_exact, 't_values': t_values}


# ═══════════════════════════════════════════════════════════════
#  7. RESOURCE ESTIMATION & CROSSOVER
# ═══════════════════════════════════════════════════════════════

def resource_estimation(max_qubits: int = 20) -> Dict:
    """
    Estimate classical vs quantum resource requirements.

    Classical: O(dim³) for diagonalization (dim = 2^{n_qubits})
    Quantum:   O(n_trotter × depth × n_qubits) for circuit

    Finds the crossover point where quantum becomes advantageous.
    """
    qubits = np.arange(2, max_qubits + 1)
    dims = 2 ** qubits

    # Classical: O(dim³) for full diagonalization
    # In practice: scipy expm scales as O(dim³) for dense matrices
    # Reference: 4-qubit (dim=16) takes ~0.01s
    t_classical_diag = 0.01 * (dims / 16) ** 3  # seconds (rough)

    # Quantum: O(n_trotter × depth × n_qubits) gates
    # On AerSimulator: ~1000 gates/shot
    # With 8192 shots and 15 Trotter steps
    n_trotter = 15
    gate_depth = n_trotter * (4 * qubits - 2)  # rough depth estimate
    shots_st = 8192
    t_quantum_aer = gate_depth * shots_st * 1e-6  # ~μs per gate-shot

    # Crossover
    crossover_idx = np.argmin(np.abs(t_classical_diag - t_quantum_aer))
    crossover_qubits = int(qubits[crossover_idx])

    return {
        'qubits': qubits.tolist(),
        'dimensions': dims.tolist(),
        'classical_time': t_classical_diag.tolist(),
        'quantum_time': t_quantum_aer.tolist(),
        'crossover_qubits': crossover_qubits,
        'crossover_time': float(t_classical_diag[crossover_idx])
    }
