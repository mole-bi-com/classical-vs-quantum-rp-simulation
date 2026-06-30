"""
Deep Project 1: Publication-Quality Figures
============================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from deep_engine import RPSystem, exact_evolution, trotter_evolution

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
})

C_EXACT = '#B2182B'
C_TROT = '#2166AC'
C_NOISE = '#4D4D4D'
C_SHOT = '#D6604D'
C_CROSS = '#762A83'


def plot_panel_a_trotter_error(t_values, P_exact, trotter_results, savepath=None):
    """Panel (a): Trotter error — exact vs increasing Trotter steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: P_S(t) curves
    ax = axes[0]
    ax.plot(t_values, P_exact, '-', color=C_EXACT, linewidth=3,
            label='Exact (matrix expm)', zorder=5)
    key_steps = [n for n in [1, 5, 10, 15, 25, 40] if n in trotter_results]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(key_steps)))
    for n, color in zip(key_steps, colors):
        ax.plot(t_values, trotter_results[n]['P_S'], '--',
                color=color, linewidth=1.5, label=f'{n} steps', alpha=0.85)
    ax.set_xlabel('Time (natural units)')
    ax.set_ylabel('$P_S(t)$')
    ax.set_title('(a) Trotterization Convergence', fontweight='bold')
    ax.legend(framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Right: Error bar chart
    ax = axes[1]
    ns = sorted(trotter_results.keys())
    mean_errs = [trotter_results[n]['mean_error'] for n in ns]
    max_errs = [trotter_results[n]['max_error'] for n in ns]
    unitary_errs = [trotter_results[n]['unitary_error'] for n in ns]

    x = np.arange(len(ns))
    w = 0.25
    ax.bar(x - w, mean_errs, w, label='Mean $|\\Delta P_S|$', color=C_TROT, alpha=0.85)
    ax.bar(x, max_errs, w, label='Max $|\\Delta P_S|$', color=C_EXACT, alpha=0.85)
    ax.bar(x + w, unitary_errs, w, label='$\\|\\Delta U\\|_F$ (norm.)', color=C_NOISE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns], fontsize=8)
    ax.set_xlabel('Number of Trotter Steps')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.set_title('(b) Error Metrics', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_panel_b_shot_noise(t_values, P_exact, shot_results, savepath=None):
    """Panel (b): Shot noise scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: P_S(t) at various shot counts
    ax = axes[0]
    ax.plot(t_values, P_exact, '-', color=C_EXACT, linewidth=3, label='Exact', zorder=5)
    key_shots = [1024, 4096, 16384, 65536]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(key_shots)))
    for shots, color in zip(key_shots, colors):
        if shots in shot_results:
            ax.plot(t_values, shot_results[shots]['P_S'], 'o--',
                    color=color, markersize=4, linewidth=1,
                    label=f'{shots} shots', alpha=0.8)
    ax.set_xlabel('Time (natural units)')
    ax.set_ylabel('$P_S(t)$')
    ax.set_title('(a) Shot Noise — 15 Trotter Steps', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: Error vs shots
    ax = axes[1]
    all_shots = sorted(shot_results.keys())
    errors = [shot_results[s]['mean_error'] for s in all_shots]
    ax.loglog(all_shots, errors, 'o-', color=C_SHOT, linewidth=2.5, markersize=8)
    # Theoretical √N scaling
    ref_shots = np.array(all_shots)
    ax.loglog(ref_shots, errors[0] * np.sqrt(ref_shots[0] / ref_shots),
              '--', color='gray', linewidth=1.5, alpha=0.6, label='$1/\\sqrt{N}$ scaling')
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Mean $|\\Delta P_S|$')
    ax.set_title('(b) Shot Noise Scaling', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_panel_c_nisq_threshold(t_values, P_exact, nisq_results, savepath=None):
    """Panel (c): NISQ noise threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Error vs Trotter steps for each noise level
    ax = axes[0]
    for label, data in nisq_results.items():
        ns = sorted([k for k in data.keys() if isinstance(k, (int, float))])
        errs = [data[n] for n in ns]
        ax.plot(ns, errs, 'o-', linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Trotter Steps')
    ax.set_ylabel('Mean $|\\Delta P_S|$')
    ax.set_yscale('log')
    ax.set_title('(a) Noise Threshold by Trotter Steps', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.4, label='5% threshold')

    # Right: Bar chart — feasible Trotter steps vs noise
    ax = axes[1]
    noise_labels = list(nisq_results.keys())
    feasible = []
    for label in noise_labels:
        data = nisq_results[label]
        ns = sorted([k for k in data.keys() if isinstance(k, (int, float))])
        below = sum(1 for n in ns if data[n] < 0.05)
        feasible.append(below)
    colors = ['#2166AC' if f > 0 else '#D6604D' for f in feasible]
    bars = ax.barh(noise_labels, feasible, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Feasible Trotter Steps (< 5% error)')
    ax.set_title('(b) NISQ Feasibility', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, feasible):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_panel_d_resource_crossover(resource_data, savepath=None):
    """Panel (d): Resource estimation & classical/quantum crossover."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Scaling comparison
    ax = axes[0]
    ax.loglog(resource_data['qubits'], resource_data['classical_time'],
              '-', color=C_EXACT, linewidth=2.5, label='Classical (diagonalization)')
    ax.loglog(resource_data['qubits'], resource_data['quantum_time'],
              '-', color=C_TROT, linewidth=2.5, label='Quantum (Trotter circuit)')
    ax.axvline(x=resource_data['crossover_qubits'], color='gray',
               linestyle='--', alpha=0.5)
    crossover_q = resource_data['crossover_qubits']
    crossover_t = resource_data['crossover_time']
    ax.annotate(f'Crossover\n~{crossover_q} qubits',
                xy=(crossover_q, crossover_t),
                xytext=(crossover_q - 4, crossover_t * 10),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Computational Time (s)')
    ax.set_title('(a) Classical vs Quantum Scaling', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    # Right: Crossover analysis
    ax = axes[1]
    qubits = np.array(resource_data['qubits'])
    classical = np.array(resource_data['classical_time'])
    quantum = np.array(resource_data['quantum_time'])
    speedup = classical / (quantum + 1e-30)
    ax.semilogy(qubits, speedup, '-', color=C_CROSS, linewidth=2.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(qubits, 1.0, speedup,
                     where=(speedup > 1), alpha=0.2, color='green', label='Quantum advantage')
    ax.fill_between(qubits, speedup, 1.0,
                     where=(speedup < 1), alpha=0.2, color='red', label='Classical advantage')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Speedup Factor (classical / quantum)')
    ax.set_title(f'(b) Crossover at ~{crossover_q} Qubits', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_publication_quad(t_values, P_exact, trotter_data,
                           shot_data, nisq_data, resource_data, savepath=None):
    """2×2 Nature-level publication figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # (a) Trotter convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_values, P_exact, '-', color=C_EXACT, linewidth=3, label='Exact', zorder=5)
    key_steps = [n for n in [1, 5, 10, 15, 25, 40] if n in trotter_data]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(key_steps)))
    for n, color in zip(key_steps, colors):
        ax1.plot(t_values, trotter_data[n]['P_S'], '--',
                 color=color, linewidth=1.5, label=f'{n} steps', alpha=0.85)
    ax1.set_xlabel('Time (nat. units)')
    ax1.set_ylabel('$P_S(t)$')
    ax1.set_title('(a) Trotter Convergence', fontweight='bold')
    ax1.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # (b) Shot noise
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_values, P_exact, '-', color=C_EXACT, linewidth=3, label='Exact', zorder=5)
    key_shots = [1024, 4096, 16384, 65536]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(key_shots)))
    for shots, color in zip(key_shots, colors):
        if shots in shot_data:
            ax2.plot(t_values, shot_data[shots]['P_S'], 'o--',
                     color=color, markersize=3, linewidth=1,
                     label=f'{shots} shots', alpha=0.8)
    ax2.set_xlabel('Time (nat. units)')
    ax2.set_ylabel('$P_S(t)$')
    ax2.set_title('(b) Shot Noise', fontweight='bold')
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # (c) NISQ threshold
    ax3 = fig.add_subplot(gs[1, 0])
    for label, data in nisq_data.items():
        ns = sorted([k for k in data.keys() if isinstance(k, (int, float))])
        errs = [data[n] for n in ns]
        ax3.plot(ns, errs, 'o-', linewidth=2, markersize=5, label=label)
    ax3.set_xlabel('Trotter Steps')
    ax3.set_ylabel('Mean $|\\Delta P_S|$')
    ax3.set_yscale('log')
    ax3.set_title('(c) NISQ Noise Threshold', fontweight='bold')
    ax3.legend(fontsize=7, framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.3)

    # (d) Resource crossover
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.loglog(resource_data['qubits'], resource_data['classical_time'],
               '-', color=C_EXACT, linewidth=2.5, label='Classical')
    ax4.loglog(resource_data['qubits'], resource_data['quantum_time'],
               '-', color=C_TROT, linewidth=2.5, label='Quantum')
    ax4.axvline(x=resource_data['crossover_qubits'], color='gray',
                linestyle='--', alpha=0.5)
    ax4.set_xlabel('Qubits')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('(d) Resource Crossover', fontweight='bold')
    ax4.legend(fontsize=8, framealpha=0.9)
    ax4.grid(True, alpha=0.3, which='both')

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)
