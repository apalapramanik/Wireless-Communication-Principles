"""
Topic 4 — Digital Signal Processing
Part A: FFT basics — decompose a composite signal into its frequencies.

Key ideas:
  - Any signal = sum of sinusoids (Fourier's theorem)
  - FFT recovers those sinusoids from a sampled signal
  - rfft is the efficient version for real-valued signals
  - Frequency resolution = f_s / N  (1 Hz per bin here)
  - Amplitude normalisation: multiply by 2/N to get true amplitudes

Run:
    python fft_basics.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Sampling parameters ──────────────────────────────────────────────
# f_s = sampling rate (how many snapshots per second)
# T   = total recording duration
# N   = total number of samples = f_s × T
# t   = time axis: [0, 1/f_s, 2/f_s, ..., (N-1)/f_s]
#
# endpoint=False: we want samples from 0 to just-before T.
# Including T would double-count the boundary when you concatenate periods.

f_s = 1000
T   = 1.0
N   = int(f_s * T)
t   = np.linspace(0, T, N, endpoint=False)


# ── Build a composite signal: three tones mixed together ─────────────
# Amplitudes 1.0, 0.5, 0.3 at 50, 120, 300 Hz.
# This is what an antenna sees: a jumble of energy at different frequencies.

f1, A1 = 50,  1.0
f2, A2 = 120, 0.5
f3, A3 = 300, 0.3

x = (A1 * np.cos(2*np.pi*f1*t) +
     A2 * np.cos(2*np.pi*f2*t) +
     A3 * np.cos(2*np.pi*f3*t))


# ── FFT ──────────────────────────────────────────────────────────────
# np.fft.rfft: real-input FFT. Returns N//2 + 1 complex numbers.
# The k-th output represents frequency k × (f_s/N).
# With N=1000, f_s=1000 → bin width = 1 Hz → bin k = frequency k Hz.
#
# np.fft.rfftfreq(N, 1/f_s) generates the frequency axis automatically.
#   d = 1/f_s = sample spacing in seconds
#   result[k] = k × f_s / N

X     = np.fft.rfft(x)
freqs = np.fft.rfftfreq(N, 1/f_s)

# Magnitude normalisation: raw |X[k]| = (N/2) × true_amplitude for a tone.
# Multiply by 2/N to recover physical amplitudes.
mag = np.abs(X) * 2 / N


# ── Identify the three strongest peaks ──────────────────────────────
top3_idx   = np.argsort(mag)[-3:][::-1]
top3_freqs = freqs[top3_idx].astype(int)
top3_amps  = mag[top3_idx]


# ── Plot ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle('FFT Basics — Composite Signal Decomposition', fontsize=13)

# Time domain: show only first 100 ms (100 samples) — cleaner view
ax1.plot(t[:100], x[:100], color='steelblue', linewidth=1)
ax1.set_title('Time Domain — first 100 ms  (looks like a messy wave)', fontsize=11)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)

# Frequency domain: stem plot shows spikes at the exact component frequencies
ax2.stem(freqs, mag, markerfmt='C1o', linefmt='C1-', basefmt='k-')
ax2.set_title('Frequency Domain — FFT reveals the three hidden tones', fontsize=11)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_xlim([0, 400])
ax2.set_ylim([0, 1.2])
ax2.axvline(f1, color='C0', ls='--', alpha=0.6, label=f'{f1} Hz  A={A1}')
ax2.axvline(f2, color='C2', ls='--', alpha=0.6, label=f'{f2} Hz  A={A2}')
ax2.axvline(f3, color='C3', ls='--', alpha=0.6, label=f'{f3} Hz  A={A3}')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_basics.png', dpi=150)
print("Plot saved → fft_basics.png")
plt.show()


# ── Print summary ─────────────────────────────────────────────────────
print()
print("=== FFT Result ===\n")
print(f"  Sampling rate : {f_s} Hz")
print(f"  Duration      : {T} s")
print(f"  N samples     : {N}")
print(f"  Bin width     : {f_s/N:.1f} Hz  (frequency resolution)")
print()
print(f"  {'Frequency (Hz)':>16}  {'Recovered Amp':>15}  {'True Amp':>10}  {'Error':>8}")
print("  " + "-" * 55)
for freq_true, amp_true in [(f1, A1), (f2, A2), (f3, A3)]:
    k         = int(freq_true * N / f_s)   # bin index for this frequency
    recovered = mag[k]
    error     = abs(recovered - amp_true)
    print(f"  {freq_true:>16}  {recovered:>15.4f}  {amp_true:>10.4f}  {error:>8.2e}")

print()
print(f"  Top 3 peaks detected at: {top3_freqs} Hz  ✅")
print()
print("  Key insight: the FFT perfectly separated three overlapping waves.")
print("  In 5G, the same operation separates hundreds of OFDM subcarriers.")
