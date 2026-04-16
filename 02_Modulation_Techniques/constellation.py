"""
Topic 3 — Modulation Schemes
Step 1: Constellation diagram generator for square M-QAM.

What this file builds:
  1. Gray code sequence  — why adjacent symbols differ by exactly 1 bit
  2. IQ grid            — the "city map" of all valid (I, Q) symbol positions
  3. Normalization      — scale so average symbol power = 1
  4. Visualization      — plot 4-QAM (QPSK), 16-QAM, 64-QAM, 256-QAM side by side

Key idea:
  Every symbol is a point in 2D space.
  I-axis (horizontal) = in-phase component
  Q-axis (vertical)   = quadrature component
  The transmitter picks one point per time slot and sends it as a radio wave.
  The receiver measures where it landed and reads the bit label off the map.

Run:
    python constellation.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Step 1 — Gray code
# ─────────────────────────────────────────────
# Gray code: a sequence where consecutive numbers differ by exactly 1 bit.
# Example for 2 bits:  00 → 01 → 11 → 10  (each step flips one bit)
# Without Gray coding: 00 → 01 → 10 → 11  (01→10 flips TWO bits — dangerous)
#
# Why it matters: when noise pushes a received point to a neighbouring symbol,
# Gray coding guarantees only 1 bit error instead of potentially log2(M) errors.

def gray_code(n_bits: int) -> list[int]:
    """
    Return the Gray code sequence for n_bits bits.
    Length = 2^n_bits.  Entry i = i XOR (i >> 1).
    """
    return [i ^ (i >> 1) for i in range(2 ** n_bits)]


# ─────────────────────────────────────────────
# Step 2 — Build the IQ grid
# ─────────────────────────────────────────────
# For square M-QAM:
#   √M points along each axis → total M points in the plane.
#   Amplitude levels: -(√M-1), -(√M-3), ..., (√M-3), (√M-1)
#   e.g. 16-QAM (√M=4): levels = [-3, -1, +1, +3]
#
# We assign Gray-coded bit labels:
#   - Low bits  (rightmost) → I-axis position
#   - High bits (leftmost)  → Q-axis position

def qam_constellation(M: int) -> tuple[np.ndarray, list[str]]:
    """
    Generate a normalised square M-QAM constellation with Gray coding.

    Args:
        M : number of constellation points (4, 16, 64, or 256)

    Returns:
        symbols    : complex array of shape (M,), average power = 1
        bit_labels : list of M bit-strings, one per symbol
    """
    assert (np.sqrt(M) % 1 == 0) and (np.log2(M) % 1 == 0), \
        "M must be a perfect-square power of 2 (4, 16, 64, 256)"

    K = int(np.sqrt(M))              # points per axis (4-QAM → K=2, 16-QAM → K=4)
    bits_per_axis = int(np.log2(K))  # bits needed to index one axis

    # Amplitude levels: odd integers symmetric around 0
    # 16-QAM → [-3, -1, +1, +3]
    levels = np.arange(-(K - 1), K, 2, dtype=float)

    gray = gray_code(bits_per_axis)  # e.g. [0, 1, 3, 2] for 2-bit axis

    symbols    = []
    bit_labels = []

    # Iterate Q-axis top-to-bottom, I-axis left-to-right
    # (standard constellation orientation)
    for q_gray_idx in gray[::-1]:      # top row first (highest Q)
        for i_gray_idx in gray:        # left column first (most negative I)
            I = levels[i_gray_idx]
            Q = levels[q_gray_idx]
            symbols.append(complex(I, Q))
            # Concatenate Q-axis bits then I-axis bits to form the full label
            label = (format(q_gray_idx, f'0{bits_per_axis}b') +
                     format(i_gray_idx, f'0{bits_per_axis}b'))
            bit_labels.append(label)

    symbols = np.array(symbols)

    # ─── Step 3 — Normalise average power to 1 ───────────────────────
    # Without this, 256-QAM symbols would have much higher power than QPSK.
    # Normalisation lets us compare BER curves on the same SNR axis.
    avg_power = np.mean(np.abs(symbols) ** 2)
    symbols  /= np.sqrt(avg_power)

    return symbols, bit_labels


# ─────────────────────────────────────────────
# Step 4 — Visualise
# ─────────────────────────────────────────────

def plot_single_constellation(M: int, ax: plt.Axes) -> None:
    symbols, labels = qam_constellation(M)
    bits_per_sym = int(np.log2(M))

    ax.scatter(symbols.real, symbols.imag,
               color='steelblue', s=60, zorder=3)

    # Label every point with its bit string (skip labels for 256-QAM — too dense)
    if M <= 64:
        for sym, lbl in zip(symbols, labels):
            ax.annotate(lbl, (sym.real, sym.imag),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=6 if M <= 16 else 5)

    ax.axhline(0, color='gray', lw=0.5, zorder=1)
    ax.axvline(0, color='gray', lw=0.5, zorder=1)
    ax.set_title(f'{M}-{"QPSK" if M == 4 else "QAM"}  '
                 f'({bits_per_sym} bits/symbol)', fontsize=11)
    ax.set_xlabel('In-phase  I', fontsize=9)
    ax.set_ylabel('Quadrature  Q', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_all_constellations() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Square QAM Constellation Diagrams — Gray Coded\n'
                 'More points = more bits/symbol = needs higher SNR',
                 fontsize=13)

    for M, ax in zip([4, 16, 64, 256], axes):
        plot_single_constellation(M, ax)

    plt.tight_layout()
    plt.savefig('constellations.png', dpi=150)
    print("Plot saved → constellations.png")
    plt.show()


# ─────────────────────────────────────────────
# Walkthrough printout
# ─────────────────────────────────────────────

def print_walkthrough() -> None:
    print("=== Gray Code — why neighbours differ by 1 bit ===\n")
    for n_bits in [1, 2, 3]:
        gc = gray_code(n_bits)
        print(f"  {n_bits}-bit Gray code: "
              + "  ".join(f"{v:0{n_bits}b}" for v in gc))
    print()

    print("=== 16-QAM: first 8 symbols (top two rows) ===\n")
    syms, labels = qam_constellation(16)
    print(f"  {'Label':<8}  {'I':>8}  {'Q':>8}  {'Power':>8}")
    print("  " + "-" * 38)
    for s, l in zip(syms[:8], labels[:8]):
        print(f"  {l:<8}  {s.real:>8.4f}  {s.imag:>8.4f}  "
              f"{abs(s)**2:>8.4f}")
    print()

    print("=== Bits per symbol vs constellation size ===\n")
    print(f"  {'M':>6}  {'bits/sym':>10}  {'Avg Power (norm)':>18}")
    print("  " + "-" * 38)
    for M in [4, 16, 64, 256]:
        s, _ = qam_constellation(M)
        bps  = int(np.log2(M))
        print(f"  {M:>6}  {bps:>10}  {np.mean(np.abs(s)**2):>18.4f}")
    print()
    print("  Average power = 1.0 for all — normalisation confirmed.")


if __name__ == "__main__":
    print_walkthrough()
    plot_all_constellations()
