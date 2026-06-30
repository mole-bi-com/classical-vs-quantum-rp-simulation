"""
======================================================================
  DEEP PROJECT 1: Classical vs Quantum — Integrated RP Comparison
======================================================================

  Research question:
    "Can NISQ quantum computers simulate radical pair spin dynamics
     better than classical computation?"

  Pipeline:
    T1: System setup + exact evolution (gold standard)
    T2: Trotterization convergence — how many steps needed?
    T3: Shot noise analysis — measurement budget
    T4: NISQ noise threshold — when does noise win?
    T5: Resource estimation — where is the crossover?
    T6: Publication-quality figure — the final answer
======================================================================
"""

import sys, os, time, warnings, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from deep_engine import (
    RPSystem, exact_evolution, trotter_evolution, nisq_evolution,
    trotter_error_analysis, shot_noise_analysis,
    nisq_noise_threshold, resource_estimation
)
from deep_visualization import (
    plot_panel_a_trotter_error, plot_panel_b_shot_noise,
    plot_panel_c_nisq_threshold, plot_panel_d_resource_crossover,
    plot_publication_quad
)

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def banner():
    print("=" * 64)
    print("  DEEP PROJECT 1: Classical vs Quantum RP Simulation")
    print("  Unified comparison engine — arXiv:2406.12986 + QuTiP")
    print("=" * 64)


def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        r = func(*args, **kwargs)
        dt = time.time() - t0
        print(f"  ⏱  {func.__name__}: {dt:.1f}s")
        return r
    return wrapper


# ═══════════════════════════════════════════════════════════════
#  Standard test system
# ═══════════════════════════════════════════════════════════════

STD_SYS = RPSystem(
    omega=0.1, J=0.01, hfc_A=[1.0], hfc_B=[0.3],
    k_S=0.5, k_T=0.5
)

T_FINAL = 50.0


@timer
def task_1_exact_baseline():
    """T1: Exact baseline — gold standard."""
    print("\n" + "=" * 50)
    print("   T1: Exact Baseline (Gold Standard)")
    print("=" * 50)
    print(f"  System: {STD_SYS}")
    print(f"  Hilbert space: {STD_SYS.dim} (2^{STD_SYS.n_qubits})")

    t_values = np.linspace(0, T_FINAL, 50)
    P_exact = exact_evolution(STD_SYS, t_values)

    print(f"  P_S(t=0)     = {P_exact[0]:.6f} (should be 1.0)")
    print(f"  P_S(t=50)    = {P_exact[-1]:.6f}")

    H = STD_SYS.build_H()
    evals = np.sort(np.real(np.linalg.eigvalsh(H)))
    print(f"  Energy spectrum: {evals[0]:.4f} … {evals[-1]:.4f}")
    print(f"  Spectral range: {evals[-1] - evals[0]:.4f}")
    return t_values, P_exact


@timer
def task_2_trotter_convergence():
    """T2: Trotterization convergence."""
    print("\n" + "=" * 50)
    print("   T2: Trotterization Convergence")
    print("=" * 50)

    result = trotter_error_analysis(STD_SYS, T_FINAL)
    r = result['results']

    print(f"  Trotter steps: {list(r.keys())}")
    print(f"  Mean errors:")
    for n, data in r.items():
        print(f"    n={n:3d}:  mean_err={data['mean_error']:.6f}  "
              f"max_err={data['max_error']:.6f}  "
              f"unitary_err={data['unitary_error']:.2e}")
    print(f"\n  Convergence rate: error ∝ n^{{-{result['convergence_rate']:.2f}}}")
    print(f"  Error at 15 steps: {result['n_15_error']:.6f} "
          f"({'✅ < 0.01' if result['n_15_error'] < 0.01 else '❌ > 0.01'})")
    print(f"  Steps needed for error < 0.01: {result['sufficient_steps']}")

    plot_panel_a_trotter_error(
        result['t_values'], result['P_exact'], r,
        savepath=os.path.join(FIG_DIR, 'panel_a_trotter.png')
    )
    return result


@timer
def task_3_shot_noise():
    """T3: Shot noise analysis."""
    print("\n" + "=" * 50)
    print("   T3: Shot Noise Analysis")
    print("=" * 50)

    result = shot_noise_analysis(STD_SYS, T_FINAL, n_trotter=15, n_time_points=30)

    for shots, data in result['results'].items():
        print(f"  {shots:6d} shots: mean_err = {data['mean_error']:.6f}")

    plot_panel_b_shot_noise(
        result['t_values'], result['P_exact'], result['results'],
        savepath=os.path.join(FIG_DIR, 'panel_b_shot.png')
    )
    return result


@timer
def task_4_nisq_threshold():
    """T4: NISQ noise threshold."""
    print("\n" + "=" * 50)
    print("   T4: NISQ Noise Threshold")
    print("=" * 50)

    result = nisq_noise_threshold(STD_SYS, T_FINAL)

    for label, data in result['results'].items():
        print(f"  {label}:")
        for n, err in data.items():
            status = '✅' if err < 0.05 else '⚠️' if err < 0.2 else '❌'
            print(f"    Trotter {n:2d}: err={err:.4f} {status}")

    plot_panel_c_nisq_threshold(
        result['t_values'], result['P_exact'], result['results'],
        savepath=os.path.join(FIG_DIR, 'panel_c_nisq.png')
    )
    return result


@timer
def task_5_resource_crossover():
    """T5: Resource estimation & crossover."""
    print("\n" + "=" * 50)
    print("   T5: Resource Crossover Analysis")
    print("=" * 50)

    result = resource_estimation(max_qubits=20)

    print(f"  Classical vs Quantum resource comparison:")
    for q, dim, t_c, t_q in zip(result['qubits'], result['dimensions'],
                                  result['classical_time'], result['quantum_time']):
        advantage = "QUANTUM" if t_q < t_c else "classical"
        print(f"    {q:2d} qubits (dim={dim:6d}):  classical={t_c:.2e}s  "
              f"quantum={t_q:.2e}s  → {advantage}")
    print(f"\n  🏆 Crossover at ~{result['crossover_qubits']} qubits "
          f"({result['crossover_time']:.2e}s)")

    plot_panel_d_resource_crossover(
        result,
        savepath=os.path.join(FIG_DIR, 'panel_d_resource.png')
    )
    return result


@timer
def task_6_publication(trotter_data, shot_data, nisq_data, resource_data):
    """T6: Publication-quality 4-panel figure."""
    print("\n" + "=" * 50)
    print("   T6: Publication Figure")
    print("=" * 50)

    t_values = trotter_data['t_values']
    P_exact = trotter_data['P_exact']

    # Reformat nisq_data for publication plot
    nisq_plot_data = {}
    for label, data in nisq_data['results'].items():
        nisq_plot_data[label] = data

    plot_publication_quad(
        t_values, P_exact,
        trotter_data['results'],
        shot_data['results'],
        nisq_plot_data,
        resource_data,
        savepath=os.path.join(FIG_DIR, 'deep1_publication.png')
    )
    print(f"  Published to: {FIG_DIR}/deep1_publication.png/.pdf")


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    banner()

    t_start = time.time()

    t1_t, t1_P = task_1_exact_baseline()
    t2 = task_2_trotter_convergence()
    t3 = task_3_shot_noise()
    t4 = task_4_nisq_threshold()
    t5 = task_5_resource_crossover()
    task_6_publication(t2, t3, t4, t5)

    total = time.time() - t_start
    print("\n" + "=" * 64)
    print(f"  ✅ ALL 6 TASKS COMPLETE — Total: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 64)

    print("\n  Figures:")
    for f in sorted(os.listdir(FIG_DIR)):
        sz = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"    {f:<45s} {sz:.1f} KB")
