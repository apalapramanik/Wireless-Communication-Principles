"""
Topic 5 — OFDM Full Transceiver
================================
Parts:
  A  QAM mapper / demapper + OFDM modulator / demodulator
  B  Multipath channel model + channel frequency response plot
  C  Full link simulation with zero-forcing equalizer + constellation plots
  D  BER vs SNR curve (with equalizer vs without, vs theoretical AWGN)

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS  (simplified 5G NR-like)
# ═══════════════════════════════════════════════════════════════════
N_FFT    = 128      # FFT size (total subcarriers)
N_CP     = 16       # cyclic prefix length (samples)
N_ACTIVE = 72       # active data subcarriers (rest are guard bands)
M        = 16       # QAM order  (16-QAM = 4 bits/symbol)
SNR_DB   = 25       # channel SNR for single-point simulation

BPS = int(np.log2(M))   # bits per symbol

# Multipath channel: 3-tap urban macro model
DELAYS = [0, 4, 9]                                          # sample delays
GAINS  = [1.0,
          0.6 * np.exp(1j * 0.8),
          0.3 * np.exp(-1j * 1.2)]                          # complex gains


# ═══════════════════════════════════════════════════════════════════
# PART A — QAM + OFDM MODULATOR / DEMODULATOR
# ═══════════════════════════════════════════════════════════════════

def qam_constellation(M):
    """
    Generate normalized M-QAM constellation points.
    Square constellation only (M must be perfect square).
    Normalized so average symbol power = 1.
    """
    m = int(np.sqrt(M))
    assert m * m == M, "M must be a perfect square"
    levels = np.arange(-(m - 1), m, 2)          # e.g. [-3,-1,1,3] for 16-QAM
    I, Q   = np.meshgrid(levels, levels)
    points = (I + 1j * Q).flatten()
    points /= np.sqrt(np.mean(np.abs(points) ** 2))   # normalize to unit power
    return points


def qam_modulate(bits, M):
    """
    Map bit stream to QAM symbols.
    Groups of log2(M) bits → one complex symbol index → constellation point.
    """
    const = qam_constellation(M)
    bps   = int(np.log2(M))
    bits  = np.array(bits)
    n_sym = len(bits) // bps
    words = bits[: n_sym * bps].reshape(-1, bps)
    idx   = np.array([int("".join(map(str, row)), 2) for row in words])
    return const[idx]


def qam_demodulate(rx_syms, M):
    """
    Nearest-neighbour hard decision demapper.
    Each received complex sample → closest constellation point → bits.
    """
    const = qam_constellation(M)
    bps   = int(np.log2(M))
    dist  = np.abs(rx_syms[:, None] - const[None, :])   # (N_sym, M) distance matrix
    idx   = np.argmin(dist, axis=1)
    bits  = []
    for i in idx:
        bits.extend([int(b) for b in format(i, f"0{bps}b")])
    return np.array(bits)


def ofdm_modulate(qam_symbols, N_fft, N_cp, N_active):
    """
    OFDM modulator:
      1. Place QAM symbols on active subcarriers (DC-centred, guard bands on edges)
      2. IFFT: frequency domain → time domain
      3. Prepend cyclic prefix
    """
    assert len(qam_symbols) == N_active
    freq_domain = np.zeros(N_fft, dtype=complex)

    # DC-centred placement: positive freqs then negative freqs
    half = N_active // 2
    freq_domain[1 : half + 1]           = qam_symbols[:half]   # positive
    freq_domain[N_fft - half : N_fft]   = qam_symbols[half:]   # negative

    time_domain = np.fft.ifft(freq_domain)     # IFFT
    cp          = time_domain[-N_cp:]          # copy tail
    return np.concatenate([cp, time_domain])   # prepend CP


def ofdm_demodulate(rx_signal, N_fft, N_cp, N_active):
    """
    OFDM demodulator:
      1. Strip cyclic prefix
      2. FFT: time domain → frequency domain
      3. Extract active subcarriers (same positions as transmitter)
    Returns: (active subcarrier values, full FFT output)
    """
    without_cp  = rx_signal[N_cp : N_cp + N_fft]
    freq_domain = np.fft.fft(without_cp)

    half    = N_active // 2
    rx_syms = np.concatenate([
        freq_domain[1 : half + 1],
        freq_domain[N_fft - half : N_fft]
    ])
    return rx_syms, freq_domain


# ── Sanity check (no channel) ──────────────────────────────────────
n_bits  = N_ACTIVE * BPS
tx_bits = np.random.randint(0, 2, n_bits)
tx_syms = qam_modulate(tx_bits, M)
tx_ofdm = ofdm_modulate(tx_syms, N_FFT, N_CP, N_ACTIVE)
rx_syms_clean, _ = ofdm_demodulate(tx_ofdm, N_FFT, N_CP, N_ACTIVE)

print("=" * 55)
print("PART A — Sanity check (no channel, no noise)")
print("=" * 55)
print(f"  Tx symbol power : {np.mean(np.abs(tx_syms)**2):.4f}")
print(f"  Rx symbol power : {np.mean(np.abs(rx_syms_clean)**2):.4f}")
print(f"  Max IQ error    : {np.max(np.abs(rx_syms_clean - tx_syms)):.2e}")
print("  Expected: max error < 1e-13 (machine precision)\n")


# ═══════════════════════════════════════════════════════════════════
# PART B — MULTIPATH CHANNEL + FREQUENCY RESPONSE PLOT
# ═══════════════════════════════════════════════════════════════════

def multipath_channel(tx, delays, gains):
    """
    Apply a multipath channel: sum of delayed, scaled copies.
    delays: list of integer sample delays
    gains : list of complex gains (amplitude + phase per path)
    Result is linear convolution — CP at receiver converts this to circular.
    """
    max_delay = max(delays)
    rx = np.zeros(len(tx) + max_delay, dtype=complex)
    for d, g in zip(delays, gains):
        rx[d : d + len(tx)] += g * tx
    return rx[: len(tx)]


def awgn(signal, snr_db):
    """Add complex AWGN noise at a given SNR (per sample)."""
    sig_pwr   = np.mean(np.abs(signal) ** 2)
    snr_lin   = 10 ** (snr_db / 10)
    noise_pwr = sig_pwr / snr_lin
    noise = np.sqrt(noise_pwr / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


print("=" * 55)
print("PART B — Multipath channel")
print("=" * 55)
print(f"  Delays (samples) : {DELAYS}")
print(f"  Max delay        : {max(DELAYS)} samples")
print(f"  CP length        : {N_CP} samples")
print(f"  CP covers delay  : {N_CP >= max(DELAYS)}\n")

# Build channel impulse response vector for plotting
h_vec = np.zeros(20, dtype=complex)
for d, g in zip(DELAYS, GAINS):
    h_vec[d] = g

H_full = np.fft.fft(h_vec, N_FFT)    # channel frequency response

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].stem(np.abs(h_vec), markerfmt="C0o", linefmt="C0-", basefmt="k-")
axes[0].axvline(N_CP, color="red", ls="--", lw=1.5, label=f"CP boundary ({N_CP})")
axes[0].set(title="Channel Impulse Response (magnitude)",
            xlabel="Delay (samples)", ylabel="|h|")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

freqs = np.fft.fftshift(np.fft.fftfreq(N_FFT))
H_shifted = np.fft.fftshift(H_full)
axes[1].plot(freqs, 20 * np.log10(np.abs(H_shifted) + 1e-10), color="steelblue")
axes[1].set(title="Channel Frequency Response",
            xlabel="Normalized frequency", ylabel="|H(f)| (dB)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Multipath Channel Model — 3 Tap Urban Macro", fontsize=12)
plt.tight_layout()
plt.savefig("04_OFDM/channel.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_OFDM/channel.png\n")


# ═══════════════════════════════════════════════════════════════════
# PART C — FULL LINK: TX → CHANNEL → RX WITH ZF EQUALIZER
# ═══════════════════════════════════════════════════════════════════

def ofdm_link(n_symbols, M, N_fft, N_cp, N_active, snr_db,
              delays, gains, use_equalizer=True):
    """
    Simulate multiple OFDM symbols through multipath + AWGN.

    Steps per symbol:
      TX:  bits → QAM → IFFT → add CP
      CH:  multipath convolution + AWGN
      RX:  strip CP → FFT → (optional) ZF equalize → QAM demod

    Returns: tx_bits, rx_bits, rx_equalized_symbols (for constellation)
    """
    bps    = int(np.log2(M))
    n_bits = n_symbols * N_active * bps
    tx_bits_all = np.random.randint(0, 2, n_bits)

    # Precompute channel frequency response at active subcarrier positions
    h_vec  = np.zeros(N_fft, dtype=complex)
    for d, g in zip(delays, gains):
        h_vec[d] = g
    H_full = np.fft.fft(h_vec, N_fft)
    half   = N_active // 2
    H_active = np.concatenate([H_full[1 : half + 1], H_full[N_fft - half : N_fft]])

    all_rx_syms = []
    rx_bits_all = []

    for i in range(n_symbols):
        # ── TX ──────────────────────────────────────────────────────
        sym_bits = tx_bits_all[i * N_active * bps : (i + 1) * N_active * bps]
        tx_syms  = qam_modulate(sym_bits, M)
        tx_ofdm  = ofdm_modulate(tx_syms, N_fft, N_cp, N_active)

        # ── Channel ─────────────────────────────────────────────────
        rx_ch    = multipath_channel(tx_ofdm, delays, gains)
        rx_noisy = awgn(rx_ch, snr_db)

        # ── RX ──────────────────────────────────────────────────────
        rx_syms, _ = ofdm_demodulate(rx_noisy, N_fft, N_cp, N_active)

        # Zero-forcing equalization: X_hat_k = Y_k / H_k
        if use_equalizer:
            rx_syms = rx_syms / H_active

        all_rx_syms.append(rx_syms)
        rx_bits_all.extend(qam_demodulate(rx_syms, M))

    return tx_bits_all, np.array(rx_bits_all), np.concatenate(all_rx_syms)


# Run both cases
N_SYM = 50

np.random.seed(42)
tx_bits, rx_bits_eq,   syms_eq   = ofdm_link(N_SYM, M, N_FFT, N_CP, N_ACTIVE,
                                               SNR_DB, DELAYS, GAINS, True)
np.random.seed(42)
_,       rx_bits_noeq, syms_noeq = ofdm_link(N_SYM, M, N_FFT, N_CP, N_ACTIVE,
                                               SNR_DB, DELAYS, GAINS, False)

n = len(tx_bits)
ber_eq   = np.sum(tx_bits != rx_bits_eq[:n])   / n
ber_noeq = np.sum(tx_bits != rx_bits_noeq[:n]) / n

print("=" * 55)
print("PART C — Full link simulation")
print("=" * 55)
print(f"  OFDM symbols simulated : {N_SYM}")
print(f"  SNR                    : {SNR_DB} dB")
print(f"  BER with ZF equalizer  : {ber_eq:.4f}")
print(f"  BER without equalizer  : {ber_noeq:.4f}")
print()

# Constellation plots
const_ref = qam_constellation(M)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(const_ref.real, const_ref.imag, s=80, color="black", zorder=5)
axes[0].set(title=f"{M}-QAM Reference (ideal)", aspect="equal")
axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(syms_noeq.real, syms_noeq.imag, s=4, alpha=0.4, color="crimson")
axes[1].scatter(const_ref.real, const_ref.imag, s=60, color="black", zorder=5, marker="x")
axes[1].set(title=f"No equalizer\nBER = {ber_noeq:.3f}", aspect="equal")
axes[1].axhline(0, color="k", lw=0.5); axes[1].axvline(0, color="k", lw=0.5)
axes[1].grid(True, alpha=0.3)

axes[2].scatter(syms_eq.real, syms_eq.imag, s=4, alpha=0.4, color="steelblue")
axes[2].scatter(const_ref.real, const_ref.imag, s=60, color="black", zorder=5, marker="x")
axes[2].set(title=f"ZF equalizer\nBER = {ber_eq:.4f}", aspect="equal")
axes[2].axhline(0, color="k", lw=0.5); axes[2].axvline(0, color="k", lw=0.5)
axes[2].grid(True, alpha=0.3)

plt.suptitle("OFDM Constellation: Effect of Multipath and ZF Equalization", fontsize=12)
plt.tight_layout()
plt.savefig("04_OFDM/ofdm_constellation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_OFDM/ofdm_constellation.png\n")


# ═══════════════════════════════════════════════════════════════════
# PART D — BER vs SNR CURVE
# ═══════════════════════════════════════════════════════════════════

def ber_theory_16qam(snr_db):
    """
    Theoretical BER for 16-QAM over AWGN (no multipath).
    Reference curve to benchmark the equalizer performance.
    """
    snr = 10 ** (snr_db / 10)
    return 0.375 * erfc(np.sqrt(0.1 * snr))


snr_range = np.arange(0, 31, 3)
bers_eq   = []
bers_noeq = []

print("=" * 55)
print("PART D — BER vs SNR sweep")
print("=" * 55)
print(f"  {'SNR (dB)':<12} {'BER (with EQ)':<18} {'BER (no EQ)'}")
print("  " + "-" * 46)

for snr in snr_range:
    np.random.seed(7)
    tx_b, rx_eq,   _ = ofdm_link(30, M, N_FFT, N_CP, N_ACTIVE,
                                   snr, DELAYS, GAINS, True)
    np.random.seed(7)
    _,    rx_noeq, _ = ofdm_link(30, M, N_FFT, N_CP, N_ACTIVE,
                                   snr, DELAYS, GAINS, False)
    nb = len(tx_b)
    b_eq   = max(np.sum(tx_b != rx_eq[:nb])   / nb, 1e-5)
    b_noeq = max(np.sum(tx_b != rx_noeq[:nb]) / nb, 1e-5)
    bers_eq.append(b_eq)
    bers_noeq.append(b_noeq)
    print(f"  {snr:<12} {b_eq:<18.4e} {b_noeq:.4e}")

snr_fine = np.linspace(0, 30, 200)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(snr_range, bers_noeq, "rs--",
            label="Multipath, NO equalizer", markersize=6, linewidth=1.5)
ax.semilogy(snr_range, bers_eq,   "bo-",
            label="Multipath + ZF equalizer", markersize=6, linewidth=1.5)
ax.semilogy(snr_fine, ber_theory_16qam(snr_fine), "k--", alpha=0.5,
            label="Theoretical 16-QAM (AWGN, no multipath)")

ax.axhline(1e-3, color="gray", ls=":", alpha=0.7, label="BER = 1e-3 target")
ax.set(xlabel="SNR (dB)", ylabel="BER",
       title="OFDM BER vs SNR — 16-QAM, 3-tap Multipath Channel",
       ylim=[1e-5, 1])
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("04_OFDM/ofdm_ber.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: 04_OFDM/ofdm_ber.png")
