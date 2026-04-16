"""
Topic 4 — Digital Signal Processing
Part C: FIR Low-Pass Filter — design, apply, and inspect.

How a windowed-sinc FIR filter is built from scratch:

  1. The ideal LPF in frequency domain is a rectangle H(f):
       H(f) = 1 for |f| ≤ cutoff,  0 otherwise

  2. Its time-domain equivalent is a sinc function (inverse FFT of rectangle):
       h_ideal[n] = 2·fc · sinc(2·fc·n)    fc = cutoff / f_s  (normalised)

  3. sinc goes on forever — truncate to n_taps coefficients.
     Multiply by a Hamming window to reduce spectral leakage at the edges.

  4. Normalise so DC gain = 1  (sum of coefficients = 1).

Applying the filter (lfilter equivalent):
     y[n] = Σ_{k=0}^{K} h[k] · x[n-k]
  = weighted average of the last K input samples.

No scipy needed — implemented entirely in numpy.

Run:
    python fir_filter.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── FIR filter design: windowed-sinc method ───────────────────────────

def firwin_lpf(n_taps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """
    Design a low-pass FIR filter using the windowed-sinc method.

    Args:
        n_taps   : number of filter coefficients (odd → symmetric, linear phase)
        cutoff_hz: cutoff frequency in Hz (3 dB point)
        fs       : sampling rate in Hz

    Returns:
        h : FIR coefficients, shape (n_taps,)
    """
    fc = cutoff_hz / fs          # normalise cutoff to [0, 0.5]

    # Centre index — the sinc is symmetric around this point
    M = n_taps - 1
    n = np.arange(n_taps)

    # Ideal sinc impulse response
    # np.sinc(x) = sin(π·x) / (π·x), so np.sinc(2·fc·(n-M/2)) gives
    # the ideal low-pass impulse response centred at M/2.
    h = 2 * fc * np.sinc(2 * fc * (n - M / 2))

    # Hamming window — tapers the sinc to reduce side-lobe ringing
    # w[n] = 0.54 - 0.46·cos(2π·n/M)
    window = np.hamming(n_taps)
    h     *= window

    # Normalise: ensure sum = 1 so a DC signal (constant) passes unchanged
    h /= h.sum()

    return h


def apply_fir(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply FIR filter h to signal x via direct convolution.
    Uses 'same' mode to keep output length = input length.

    y[n] = Σ_{k} h[k] · x[n-k]
    """
    return np.convolve(x, h, mode='same')


def freq_response(h: np.ndarray, fs: float,
                  n_points: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency response of filter h.
    Zero-pads h to n_points before FFT for smooth curve.

    Returns:
        freqs : frequency axis in Hz
        H     : complex frequency response
    """
    H     = np.fft.rfft(h, n=n_points)
    freqs = np.fft.rfftfreq(n_points, 1 / fs)
    return freqs, H


# ── Build and apply the filter ────────────────────────────────────────

f_s    = 1000
cutoff = 80      # Hz — pass tones below 80 Hz, block above
n_taps = 101     # odd → symmetric → linear phase (no distortion)

h = firwin_lpf(n_taps, cutoff, f_s)

# Signal: 50 Hz (wanted) + 200 Hz (interference to remove)
t = np.linspace(0, 1.0, f_s, endpoint=False)
x = np.cos(2*np.pi*50*t) + 0.8*np.cos(2*np.pi*200*t)
y = apply_fir(h, x)

# Frequency response
freqs_h, H = freq_response(h, f_s)
H_db       = 20 * np.log10(np.abs(H) + 1e-12)   # convert to dB


# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle('FIR Low-Pass Filter  (windowed-sinc, Hamming window)\n'
             f'n_taps={n_taps}, cutoff={cutoff} Hz, f_s={f_s} Hz', fontsize=13)

# 1. Filter frequency response
axes[0].plot(freqs_h, H_db, color='steelblue', linewidth=2)
axes[0].axvline(cutoff, color='red', ls='--', linewidth=1.5,
                label=f'Cutoff = {cutoff} Hz')
axes[0].axhline(-3,  color='gray', ls=':', alpha=0.7, label='-3 dB point')
axes[0].axhline(-40, color='orange', ls=':', alpha=0.7, label='-40 dB stopband')
axes[0].set_title('Filter Frequency Response', fontsize=11)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_ylim([-80, 5])
axes[0].set_xlim([0, 500])
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. Input signal (time domain, first 300 samples)
axes[1].plot(t[:300], x[:300], color='crimson', linewidth=1)
axes[1].set_title('Input:  50 Hz wanted + 200 Hz interference', fontsize=11)
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

# 3. Output signal — 200 Hz should be gone, 50 Hz intact
axes[2].plot(t[:300], y[:300], color='forestgreen', linewidth=1)
axes[2].set_title('Output: 200 Hz removed, 50 Hz passed through', fontsize=11)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fir_filter.png', dpi=150)
print("Plot saved → fir_filter.png")
plt.show()


# ── Numerical check ───────────────────────────────────────────────────
# FFT the output and measure how much of each tone survived
Y     = np.fft.rfft(y)
freqs = np.fft.rfftfreq(len(y), 1 / f_s)
mag_y = np.abs(Y) * 2 / len(y)

idx_50  = np.argmin(np.abs(freqs - 50))
idx_200 = np.argmin(np.abs(freqs - 200))

print()
print("=== FIR Filter Verification ===\n")
print(f"  {'Frequency':>12}  {'Input Amp':>12}  {'Output Amp':>12}  {'Attenuation':>14}")
print("  " + "-" * 54)
print(f"  {'50 Hz':>12}  {'1.000':>12}  {mag_y[idx_50]:>12.4f}  "
      f"{'PASSED ✅':>14}")
print(f"  {'200 Hz':>12}  {'0.800':>12}  {mag_y[idx_200]:>12.4f}  "
      f"{'BLOCKED ✅' if mag_y[idx_200] < 0.01 else 'PARTIAL ⚠️':>14}")
print()

# Filter coefficient summary
print("=== Filter Coefficients (first 11 of 101) ===\n")
print("  " + "  ".join(f"{v:+.4f}" for v in h[:11]) + "  ...")
print()
print(f"  Sum of all taps : {h.sum():.6f}  (should be 1.0 — DC gain = 1)")
print(f"  Number of taps  : {n_taps}")
print(f"  Group delay     : {(n_taps-1)//2} samples = "
      f"{(n_taps-1)//2 / f_s * 1000:.1f} ms  (linear phase → constant delay)")
print()
print("  5G relevance: pulse-shaping filters in the TX chain use the same")
print("  windowed-sinc design to limit the signal to its allocated bandwidth.")
