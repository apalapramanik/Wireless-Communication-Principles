"""
Topic 4 — Digital Signal Processing
Part B: Aliasing — what happens when you sample too slowly.

Nyquist-Shannon theorem:  f_s  ≥  2 × f_max

If violated, high-frequency content "folds back" and masquerades
as a lower frequency.  The alias frequency is:

    f_alias = | f_signal - round(f_signal / f_s) × f_s |

Example: f_signal=100 Hz, f_s=150 Hz
    round(100/150) = 1
    f_alias = |100 - 1×150| = 50 Hz  ← the ADC thinks it's seeing 50 Hz!

In 5G, the ADC must run at ≥ 2× the channel bandwidth (e.g. ≥ 200 Msps
for a 100 MHz channel) to avoid this exact problem.

Run:
    python aliasing.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


f_signal = 100      # Hz — the true signal we want to capture
t_fine   = np.linspace(0, 0.05, 10_000)    # high-res "ground truth" for plotting


def alias_frequency(f: float, f_s: float) -> float:
    """
    Compute the alias frequency when signal f is sampled at rate f_s.
    Returns f itself if f_s >= 2*f (no aliasing).
    """
    if f_s >= 2 * f:
        return f
    return abs(f - round(f / f_s) * f_s)


# ── Three sampling scenarios ──────────────────────────────────────────
# 1. f_s = 1000 Hz → well above Nyquist  (f_s >> 2×100)  ✅
# 2. f_s = 210  Hz → just above Nyquist  (f_s > 2×100)   ✅ (barely)
# 3. f_s = 150  Hz → below Nyquist       (f_s < 2×100)   ❌ aliased

sampling_rates = [1000, 210, 150]

fig, axes = plt.subplots(3, 1, figsize=(11, 9))
fig.suptitle('Aliasing Demo — Nyquist Theorem in Action\n'
             f'True signal: {f_signal} Hz cosine', fontsize=13)

for ax, f_s in zip(axes, sampling_rates):
    # Sample the true signal at this rate
    t_s = np.arange(0, 0.05, 1 / f_s)
    x_s = np.cos(2 * np.pi * f_signal * t_s)

    # Find the dominant frequency the sampler perceives via FFT
    X    = np.fft.rfft(x_s, n=len(t_s))
    freq = np.fft.rfftfreq(len(t_s), 1 / f_s)
    peak = freq[np.argmax(np.abs(X))]

    nyquist   = f_s / 2
    ok        = f_s >= 2 * f_signal
    status    = '✅ OK' if ok else '❌ ALIASED'
    alias_f   = alias_frequency(f_signal, f_s)

    # Plot the true continuous signal in the background
    ax.plot(t_fine, np.cos(2 * np.pi * f_signal * t_fine),
            color='steelblue', alpha=0.25, linewidth=1, label='True 100 Hz signal')

    # Plot the actual sample points
    ax.plot(t_s, x_s, 'ro-', markersize=5, linewidth=1.2,
            label=f'Samples  ({len(t_s)} points)')

    # If aliased, show what the sampler thinks it sees
    if not ok:
        ax.plot(t_fine, np.cos(2 * np.pi * alias_f * t_fine),
                color='orange', alpha=0.7, linewidth=1.5,
                label=f'Perceived alias: {alias_f:.0f} Hz')

    ax.set_title(
        f'f_s = {f_s} Hz   |   Nyquist limit = {nyquist:.0f} Hz   |   '
        f'Perceives {peak:.0f} Hz   {status}',
        fontsize=10
    )
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.4, 1.6])

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('aliasing.png', dpi=150)
print("Plot saved → aliasing.png")
plt.show()


# ── Numerical summary ─────────────────────────────────────────────────
print()
print("=== Aliasing Summary ===\n")
print(f"  True signal: {f_signal} Hz\n")
print(f"  {'f_s (Hz)':>10}  {'Nyquist (Hz)':>14}  "
      f"{'Nyquist OK?':>12}  {'Perceived freq':>16}  {'Alias formula'}")
print("  " + "-" * 72)
for f_s in [1000, 500, 210, 150, 120]:
    nyq    = f_s / 2
    ok     = f_s >= 2 * f_signal
    alias  = alias_frequency(f_signal, f_s)
    formula = f"|{f_signal} - {round(f_signal/f_s)}×{f_s}| = {alias:.0f}" if not ok else "—"
    print(f"  {f_s:>10}  {nyq:>14.0f}  "
          f"{'✅' if ok else '❌':>12}  {alias:>16.0f}  {formula}")

print()
print("  5G NR relevance: a 100 MHz channel requires ADC ≥ 200 Msps.")
print("  mmWave 400 MHz channels → ADC must run at ≥ 800 Msps.")
