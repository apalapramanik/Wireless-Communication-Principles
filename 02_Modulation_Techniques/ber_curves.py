"""
Topic 3 — Modulation Schemes
Step 3: BER vs SNR curves — theory vs simulation.

Theory formula for square M-QAM over AWGN (Gray coded):

    BER ≈ (4/log₂M) × (1 − 1/√M) × Q(√(3·log₂M·Eb/N0 / (M−1)))

    where Q(x) = 0.5 × erfc(x / √2)
    and   Eb/N0 = SNR_sym / log₂M

What the plot shows:
  - Lines = theoretical BER (exact formula)
  - Dots  = simulated BER from transceiver.py
  - They overlap — validating the simulation
  - Higher M → curve shifts RIGHT (needs more SNR for same BER)

5G NR target BER: 10⁻³ before HARQ, ~10⁻⁵ after channel coding.

Run:
    python ber_curves.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transceiver import simulate_ber


# ─────────────────────────────────────────────────────────────────────
# Q-function and theoretical BER
# ─────────────────────────────────────────────────────────────────────
# Q(x) = P(Z > x) for Z ~ N(0,1).
# It answers: what fraction of noise samples are large enough to push
# a received point past the decision boundary into a wrong symbol?
#
# scipy's erfc(x) = 1 − erf(x) = 2·Q(x·√2), so Q(x) = 0.5·erfc(x/√2).

def q_func(x: np.ndarray) -> np.ndarray:
    return np.array([0.5 * math.erfc(v / math.sqrt(2)) for v in np.atleast_1d(x)])


def ber_theory(M: int, snr_db_range: np.ndarray) -> np.ndarray:
    """
    Theoretical BER for square M-QAM, AWGN, Gray coded.
    snr_db_range is SNR per symbol in dB.
    """
    snr_lin = 10 ** (snr_db_range / 10)
    k       = np.log2(M)
    eb_n0   = snr_lin / k

    # How many noise std-devs separate adjacent decision boundaries.
    # Larger → points further apart → lower BER.
    arg = np.sqrt(3 * k * eb_n0 / (M - 1))
    return (4 / k) * (1 - 1 / np.sqrt(M)) * q_func(arg)


def snr_for_ber(M: int, target_ber: float = 1e-3) -> float:
    """Binary search: SNR (dB) where theoretical BER = target_ber."""
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ber_theory(M, np.array([mid]))[0] > target_ber:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────

def plot_ber_curves() -> None:
    snr_range = np.arange(0, 37, 0.5)
    configs = [
        (4,   'QPSK',    'tab:blue',   range(0,  22, 2)),
        (16,  '16-QAM',  'tab:orange', range(4,  26, 2)),
        (64,  '64-QAM',  'tab:green',  range(8,  32, 2)),
        (256, '256-QAM', 'tab:red',    range(14, 38, 2)),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for M, label, color, sim_snrs in configs:
        ber_th = ber_theory(M, snr_range)
        ax.semilogy(snr_range, ber_th, '-', color=color,
                    linewidth=2, label=f'{label} — theory')

        sim_bers = [max(simulate_ber(M, snr_db=s, n_symbols=100_000), 1e-7)
                    for s in sim_snrs]
        ax.semilogy(list(sim_snrs), sim_bers, 'o',
                    color=color, markersize=5, label=f'{label} — sim')

    ax.axhline(1e-3, color='gray', linestyle='--', alpha=0.6,
               label='BER = 10⁻³  (5G pre-coding target)')
    ax.axhline(1e-5, color='gray', linestyle=':',  alpha=0.6,
               label='BER = 10⁻⁵  (after channel coding)')

    ax.set_xlabel('SNR per symbol (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs SNR — Square QAM, AWGN, Gray Coded\n'
                 'Lines = theory   |   Dots = simulation', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-7, 1])
    ax.set_xlim([0, 37])
    plt.tight_layout()
    plt.savefig('ber_curves.png', dpi=150)
    print("Plot saved → ber_curves.png")
    plt.show()


if __name__ == "__main__":
    print("=== SNR required to achieve BER = 10⁻³ (theory) ===\n")
    print(f"  {'Scheme':<10}  {'bits/sym':>10}  {'Required SNR (dB)':>20}")
    print("  " + "-" * 44)
    for M, name in [(4,'QPSK'), (16,'16-QAM'), (64,'64-QAM'), (256,'256-QAM')]:
        snr_req = snr_for_ber(M, target_ber=1e-3)
        print(f"  {name:<10}  {int(np.log2(M)):>10}  {snr_req:>20.1f}")
    print()
    print("  Each step up costs ~6–7 dB more SNR but adds 2 bits/symbol.")
    print()

    print("=== Theoretical BER at key SNR points ===\n")
    snr_points = [5, 10, 15, 20, 25, 30]
    header = "  {:>6}".format("SNR") + "".join(
        f"  {m:>12}" for m in ['QPSK', '16-QAM', '64-QAM', '256-QAM'])
    print(header)
    print("  " + "-" * (6 + 4 * 14))
    for snr in snr_points:
        row = f"  {snr:>6}"
        for M in [4, 16, 64, 256]:
            b = ber_theory(M, np.array([float(snr)]))[0]
            row += f"  {b:>12.2e}"
        print(row)
    print()

    plot_ber_curves()
