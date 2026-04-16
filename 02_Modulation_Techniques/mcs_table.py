"""
Topic 3 — Modulation Schemes
Step 4: 5G NR MCS table and spectral efficiency visualisation.

The MCS (Modulation and Coding Scheme) table maps a CQI index to:
  - Modulation order  (QPSK / 16-QAM / 64-QAM)
  - Code rate         (fraction of bits that are data, not redundancy)
  - Spectral efficiency in bits/s/Hz

The gNB reads the UE's CQI report, looks up this table, and picks the
highest MCS whose required SNR is ≤ the measured channel SNR.
This is Adaptive Modulation & Coding (AMC).

Source: 3GPP TS 38.214 Table 5.1.3.1-1 (64-QAM table).

Run:
    python mcs_table.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# 5G NR MCS table (3GPP TS 38.214, Table 5.1.3.1-1)
# Columns: (MCS index, modulation order Q_m, code rate × 1024, spectral efficiency)
# Q_m: 2 = QPSK, 4 = 16-QAM, 6 = 64-QAM
# ─────────────────────────────────────────────────────────────────────

MCS_TABLE = [
    (0,  2, 120,  0.2344),
    (1,  2, 157,  0.3066),
    (2,  2, 193,  0.3770),
    (3,  2, 251,  0.4902),
    (4,  2, 308,  0.6016),
    (5,  2, 379,  0.7402),
    (6,  2, 449,  0.8770),
    (7,  2, 526,  1.0273),
    (8,  2, 602,  1.1758),
    (9,  2, 679,  1.3262),
    (10, 4, 340,  1.3281),
    (11, 4, 378,  1.4766),
    (12, 4, 434,  1.6953),
    (13, 4, 490,  1.9141),
    (14, 4, 553,  2.1602),
    (15, 4, 616,  2.4063),
    (16, 4, 658,  2.5703),
    (17, 6, 438,  3.2227),
    (18, 6, 466,  3.4277),
    (19, 6, 517,  3.8086),
    (20, 6, 567,  4.1602),
    (21, 6, 616,  4.5234),
    (22, 6, 666,  4.8867),
    (23, 6, 719,  5.2734),
    (24, 6, 772,  5.6602),
    (25, 6, 822,  6.0234),
    (26, 6, 873,  6.3984),
    (27, 6, 910,  6.6602),
    (28, 6, 948,  6.9141),
]

MOD_NAME  = {2: 'QPSK', 4: '16-QAM', 6: '64-QAM'}
MOD_COLOR = {2: 'tab:blue', 4: 'tab:orange', 6: 'tab:green'}


def throughput_mbps(mcs_idx: int, bandwidth_hz: float,
                    n_resource_blocks: int = 52) -> float:
    """
    Estimate peak throughput for one MCS on a given channel bandwidth.

    Uses the simplified formula:
        Throughput = SE × BW × overhead_factor

    overhead_factor ≈ 0.75 accounts for control channels, reference signals,
    guard bands, and cyclic prefix overhead.
    """
    _, _, _, se = MCS_TABLE[mcs_idx]
    return se * bandwidth_hz * 0.75 / 1e6   # Mbps


def print_mcs_table() -> None:
    print("=== 5G NR MCS Table (3GPP TS 38.214, Table 5.1.3.1-1) ===\n")
    print(f"  {'MCS':>4}  {'Modulation':>10}  {'Code Rate':>10}  "
          f"{'SE (bps/Hz)':>12}  {'~Tput @100MHz':>16}")
    print("  " + "-" * 58)
    for mcs, qm, cr, se in MCS_TABLE:
        name  = MOD_NAME[qm]
        tput  = throughput_mbps(mcs, 100e6)
        print(f"  {mcs:>4}  {name:>10}  {cr/1024:>10.4f}  "
              f"{se:>12.4f}  {tput:>13.1f} Mbps")
    print()
    print("  Code rate = fraction of bits that carry data (rest = error correction).")
    print("  SE = spectral efficiency = bits delivered per second per Hz of bandwidth.")


def plot_spectral_efficiency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('5G NR MCS Table — Spectral Efficiency & Throughput\n'
                 '(3GPP TS 38.214 Table 5.1.3.1-1, 64-QAM table)', fontsize=13)

    indices = [row[0] for row in MCS_TABLE]
    se_vals = [row[3] for row in MCS_TABLE]
    colors  = [MOD_COLOR[row[1]] for row in MCS_TABLE]

    # Left: SE bar chart
    ax = axes[0]
    bars = ax.bar(indices, se_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('MCS Index', fontsize=11)
    ax.set_ylabel('Spectral Efficiency (bps/Hz)', fontsize=11)
    ax.set_title('Spectral Efficiency per MCS', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MOD_COLOR[q], label=MOD_NAME[q])
                       for q in [2, 4, 6]]
    ax.legend(handles=legend_elements, fontsize=10)

    # Modulation boundary lines
    ax.axvline(x=9.5,  color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=16.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(4,  6.5, 'QPSK',    ha='center', fontsize=9, color='tab:blue')
    ax.text(13, 6.5, '16-QAM',  ha='center', fontsize=9, color='tab:orange')
    ax.text(23, 6.5, '64-QAM',  ha='center', fontsize=9, color='tab:green')

    # Right: estimated throughput at different bandwidths
    ax2 = axes[1]
    bw_configs = [(20e6,  '20 MHz (LTE)'),
                  (100e6, '100 MHz (5G FR1)'),
                  (400e6, '400 MHz (5G mmWave)')]

    for bw, bw_label in bw_configs:
        tputs = [throughput_mbps(m, bw) for m, *_ in MCS_TABLE]
        ax2.plot(indices, tputs, marker='o', markersize=3, label=bw_label)

    ax2.set_xlabel('MCS Index', fontsize=11)
    ax2.set_ylabel('Estimated Throughput (Mbps)', fontsize=11)
    ax2.set_title('Peak Throughput vs MCS Index\n(overhead factor = 0.75)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=9.5,  color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=16.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('mcs_table.png', dpi=150)
    print("Plot saved → mcs_table.png")
    plt.show()


if __name__ == "__main__":
    print_mcs_table()
    print()

    # Show the AMC decision logic
    print("=== AMC: which MCS would the gNB pick? ===\n")
    print("  (Approximate SNR thresholds from BER = 10⁻³ requirement)\n")

    # Rough SNR thresholds per modulation + code rate (illustrative)
    snr_thresholds = [
        (0,  -6.7), (1,  -4.7), (2,  -2.3), (3,   0.2), (4,   2.4),
        (5,   4.3), (6,   5.9), (7,   7.7), (8,   9.0), (9,  10.3),
        (10, 11.2), (11, 12.2), (12, 13.7), (13, 15.1), (14, 16.3),
        (15, 17.7), (16, 18.7), (17, 21.2), (18, 22.7), (19, 24.2),
        (20, 25.7), (21, 27.3), (22, 28.3), (23, 29.7), (24, 31.2),
        (25, 32.2), (26, 33.4), (27, 34.3), (28, 35.2),
    ]

    test_snrs = [-5, 0, 5, 10, 15, 20, 25, 30, 35]
    print(f"  {'Measured SNR':>14}  {'Best MCS':>9}  "
          f"{'Modulation':>10}  {'SE':>8}  {'~Tput @100MHz':>16}")
    print("  " + "-" * 62)
    for snr in test_snrs:
        best_mcs = 0
        for mcs_i, thresh in snr_thresholds:
            if snr >= thresh:
                best_mcs = mcs_i
        _, qm, _, se = MCS_TABLE[best_mcs]
        tput = throughput_mbps(best_mcs, 100e6)
        print(f"  {snr:>12} dB  {best_mcs:>9}  "
              f"{MOD_NAME[qm]:>10}  {se:>8.4f}  {tput:>13.1f} Mbps")

    print()
    print("  This is exactly what happens inside your phone every ~10ms.")
    print("  The UE measures SNR, reports CQI, gNB selects MCS, repeat.")
    print()
    plot_spectral_efficiency()
