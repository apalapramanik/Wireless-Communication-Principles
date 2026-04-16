"""
Topic 3 — Modulation Schemes
Step 2: Full baseband transmitter / receiver over an AWGN channel.

Pipeline:
    random bits
        │
        ▼ TX: group bits → look up symbol in constellation
    IQ symbols
        │
        ▼ CHANNEL: add complex Gaussian noise (AWGN)
    noisy IQ symbols
        │
        ▼ RX: find nearest constellation point (minimum Euclidean distance)
    detected symbols → bits
        │
        ▼ compare with TX bits → count errors → BER

Key concepts explained in-line below.

Run:
    python transceiver.py
"""

import numpy as np
from constellation import qam_constellation


# ─────────────────────────────────────────────────────────────────────
# The transmitter
# ─────────────────────────────────────────────────────────────────────
# Job: turn a stream of random bits into a sequence of complex symbols.
#
# Example for 16-QAM (4 bits/symbol):
#   bits:    1 0 1 1  0 0 1 0  1 1 0 0  ...
#   group:   [1011]   [0010]   [1100]   ...
#   index:   11       2        12       ... (decimal value of bit group)
#   symbol:  look up constellation[11], constellation[2], ...

def transmit(bits: np.ndarray, constellation: np.ndarray,
             bits_per_sym: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Map bits to complex symbols.

    Returns:
        tx_symbols : complex array, one symbol per group
        tx_indices : integer array, index into constellation per group
    """
    n_symbols  = len(bits) // bits_per_sym
    bit_matrix = bits[: n_symbols * bits_per_sym].reshape(n_symbols, bits_per_sym)

    # Convert each row of bits to its decimal index
    # e.g. [1, 0, 1, 1] → 11
    powers     = 2 ** np.arange(bits_per_sym - 1, -1, -1)
    tx_indices = (bit_matrix * powers).sum(axis=1).astype(int)
    tx_symbols = constellation[tx_indices]
    return tx_symbols, tx_indices


# ─────────────────────────────────────────────────────────────────────
# The AWGN channel
# ─────────────────────────────────────────────────────────────────────
# AWGN = Additive White Gaussian Noise.
# "Additive"  — noise is simply added to the signal.
# "White"     — equal power at all frequencies (flat spectrum).
# "Gaussian"  — noise amplitude follows a normal distribution.
#
# The noise is complex: both I and Q components are independently noisy.
# Noise variance per component = σ² = 1 / (2 × SNR_linear).
#
# Why SNR_linear / 2?  Because SNR = signal power / (noise power per dim × 2).
# Since constellation is normalised (avg power = 1), signal power = 1.

def awgn_channel(symbols: np.ndarray, snr_db: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Pass symbols through an AWGN channel.

    Args:
        symbols : complex TX symbols (avg power = 1 assumed)
        snr_db  : signal-to-noise ratio in dB (per symbol)
        rng     : numpy random Generator for reproducibility

    Returns:
        rx_symbols : noisy received symbols
    """
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(0.5 / snr_lin)   # per I or Q component

    noise = noise_std * (rng.standard_normal(len(symbols)) +
                         1j * rng.standard_normal(len(symbols)))
    return symbols + noise


# ─────────────────────────────────────────────────────────────────────
# The receiver — minimum Euclidean distance detector
# ─────────────────────────────────────────────────────────────────────
# The receiver doesn't know which symbol was sent.
# It measures the received (I, Q) point and asks:
#   "Which constellation point is closest to what I received?"
#
# This is optimal for AWGN (maximum likelihood detection = nearest neighbour).
#
# Implementation trick: compute distance from every received point to every
# constellation point in one vectorised matrix operation, then take argmin.

def receive(rx_symbols: np.ndarray,
            constellation: np.ndarray) -> np.ndarray:
    """
    Minimum Euclidean distance detector.

    Returns:
        rx_indices : index of the nearest constellation point for each symbol
    """
    # Shape: (n_symbols, M)
    # distances[i, j] = |rx_symbols[i] - constellation[j]|
    distances  = np.abs(rx_symbols[:, None] - constellation[None, :])
    return np.argmin(distances, axis=1)


# ─────────────────────────────────────────────────────────────────────
# BER calculation
# ─────────────────────────────────────────────────────────────────────
# Compare TX bit matrix with RX bit matrix, count mismatches.

def compute_ber(tx_indices: np.ndarray, rx_indices: np.ndarray,
                bit_labels: list[str], bits_per_sym: int) -> float:
    n_symbols = len(tx_indices)
    errors    = 0
    for tx_i, rx_i in zip(tx_indices, rx_indices):
        tx_label = bit_labels[tx_i]
        rx_label = bit_labels[rx_i]
        errors  += sum(a != b for a, b in zip(tx_label, rx_label))
    return errors / (n_symbols * bits_per_sym)


# ─────────────────────────────────────────────────────────────────────
# Full pipeline — convenience wrapper
# ─────────────────────────────────────────────────────────────────────

def simulate_ber(M: int, snr_db: float,
                 n_symbols: int = 50_000,
                 seed: int = 42) -> float:
    """Run the full TX → channel → RX pipeline and return BER."""
    bits_per_sym       = int(np.log2(M))
    constellation, labels = qam_constellation(M)
    rng                = np.random.default_rng(seed)

    tx_bits            = rng.integers(0, 2, n_symbols * bits_per_sym)
    tx_symbols, tx_idx = transmit(tx_bits, constellation, bits_per_sym)
    rx_symbols         = awgn_channel(tx_symbols, snr_db, rng)
    rx_idx             = receive(rx_symbols, constellation)

    return compute_ber(tx_idx, rx_idx, labels, bits_per_sym)


# ─────────────────────────────────────────────────────────────────────
# Scatter plot — see the noise cloud around each constellation point
# ─────────────────────────────────────────────────────────────────────

def plot_noisy_constellation(M: int, snr_db: float,
                             n_symbols: int = 3_000) -> None:
    import matplotlib.pyplot as plt

    bits_per_sym          = int(np.log2(M))
    constellation, labels = qam_constellation(M)
    rng                   = np.random.default_rng(0)

    tx_bits               = rng.integers(0, 2, n_symbols * bits_per_sym)
    tx_syms, _            = transmit(tx_bits, constellation, bits_per_sym)
    rx_syms               = awgn_channel(tx_syms, snr_db, rng)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, syms, title in [
        (axes[0], tx_syms, f'{M}-QAM  TX (no noise)'),
        (axes[1], rx_syms, f'{M}-QAM  RX after AWGN  SNR={snr_db} dB'),
    ]:
        ax.scatter(syms.real, syms.imag, s=4, alpha=0.4, color='steelblue')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'noisy_constellation_{M}qam_{snr_db}dB.png', dpi=150)
    print(f"Plot saved → noisy_constellation_{M}qam_{snr_db}dB.png")
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    print("=== BER at SNR = 20 dB ===\n")
    print(f"  {'M':>6}  {'bits/sym':>10}  {'BER':>12}")
    print("  " + "-" * 32)
    for M in [4, 16, 64, 256]:
        ber = simulate_ber(M, snr_db=20)
        print(f"  {M:>6}  {int(np.log2(M)):>10}  {ber:.3e}")

    print()
    print("  Observation: same SNR, higher-order QAM = higher BER.")
    print("  256-QAM needs ~25 dB more SNR than QPSK for the same BER.")
    print()

    print("=== BER vs SNR for 16-QAM ===\n")
    print(f"  {'SNR (dB)':>10}  {'BER':>12}")
    print("  " + "-" * 25)
    for snr in [5, 10, 15, 20, 25, 30]:
        ber = simulate_ber(16, snr_db=snr)
        bar = '█' * max(1, int(-np.log10(max(ber, 1e-7))))
        print(f"  {snr:>10}  {ber:.3e}  {bar}")

    print()
    print("=== Step-by-step trace — 16-QAM, 10 symbols ===\n")
    M              = 16
    bits_per_sym   = 4
    constellation, labels = qam_constellation(M)
    rng            = np.random.default_rng(7)
    tx_bits        = rng.integers(0, 2, 10 * bits_per_sym)
    tx_syms, tx_i  = transmit(tx_bits, constellation, bits_per_sym)
    rx_syms        = awgn_channel(tx_syms, snr_db=15, rng=rng)
    rx_i           = receive(rx_syms, constellation)

    print(f"  {'TX bits':<10}  {'TX (I,Q)':>18}  {'RX (I,Q)':>18}  "
          f"{'RX bits':<10}  {'OK?'}")
    print("  " + "-" * 70)
    for t, tx, rx, r in zip(tx_i, tx_syms, rx_syms, rx_i):
        ok = '✅' if t == r else '❌'
        print(f"  {labels[t]:<10}  ({tx.real:+.3f},{tx.imag:+.3f})  "
              f"  ({rx.real:+.3f},{rx.imag:+.3f})  {labels[r]:<10}  {ok}")

    print()
    plot_noisy_constellation(16, snr_db=15)
    plot_noisy_constellation(64, snr_db=25)
