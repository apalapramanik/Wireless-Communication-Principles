"""
Topic 4 — Digital Signal Processing
Part D: Spectrogram — time + frequency together (Short-Time Fourier Transform).

The FFT gives you ONE frequency snapshot of the entire signal.
The spectrogram gives you frequency content at every moment in time.

How it works (STFT — Short-Time Fourier Transform):
  1. Slide a window of width nperseg across the signal, stepping by (nperseg - noverlap)
  2. At each position, compute the FFT of that window
  3. Stack all the FFT magnitude columns → 2D image:
       X-axis = time,  Y-axis = frequency,  colour = power

Time vs frequency resolution tradeoff (Heisenberg-Gabor limit):
  Δt × Δf ≥ 1 / (4π)
  Short window → fine time resolution, coarse frequency resolution
  Long window  → coarse time resolution, fine frequency resolution

In 5G, this tradeoff appears in OFDM symbol design:
  subcarrier spacing (Δf) × symbol duration (Δt) = 1

No scipy needed — implemented entirely in numpy.

Run:
    python spectrogram.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── STFT / Spectrogram from scratch ───────────────────────────────────

def stft(x: np.ndarray, fs: float,
         nperseg: int = 128,
         noverlap: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform.

    Args:
        x        : input signal (1D)
        fs       : sampling rate (Hz)
        nperseg  : samples per window (controls frequency resolution)
        noverlap : overlap between consecutive windows (controls time resolution)

    Returns:
        freqs   : frequency axis (Hz),  shape (nperseg//2 + 1,)
        times   : time axis (s),        shape (n_frames,)
        Sxx     : power spectral density, shape (len(freqs), n_frames)
    """
    step    = nperseg - noverlap
    n_frames = (len(x) - nperseg) // step + 1

    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    times = np.arange(n_frames) * step / fs

    # Hanning window: tapers each segment to zero at edges,
    # reducing spectral leakage between adjacent frequency bins.
    window = np.hanning(nperseg)

    Sxx = np.zeros((len(freqs), n_frames))

    for i in range(n_frames):
        start   = i * step
        segment = x[start : start + nperseg] * window
        fft_out = np.fft.rfft(segment)
        Sxx[:, i] = np.abs(fft_out) ** 2   # power = |amplitude|²

    return freqs, times, Sxx


# ── Chirp signal ───────────────────────────────────────────────────────
# A chirp is a sinusoid whose frequency changes over time.
# Here: starts at 50 Hz, sweeps linearly to 400 Hz over 2 seconds.
#
# Instantaneous frequency:  f(t) = 50 + 175·t
# To get the signal:  x(t) = cos(2π · ∫f(t)dt) = cos(2π · (50t + 87.5t²))
#
# np.cumsum approximates the integral: ∫f(t)dt ≈ Σ f[n]/f_s

f_s = 1000
t   = np.linspace(0, 2.0, 2 * f_s, endpoint=False)

f_inst = 50 + 175 * t           # instantaneous frequency at each time step
phase  = np.cumsum(f_inst) / f_s # ∫ f(t) dt  (numerical integration)
x_chirp = np.cos(2 * np.pi * phase)


# ── Compute spectrogram ───────────────────────────────────────────────
freqs, times, Sxx = stft(x_chirp, f_s, nperseg=128, noverlap=96)
Sxx_db = 10 * np.log10(Sxx + 1e-12)   # convert to dB for better visual contrast


# ── Also demo two pure tones switching at t=1s ────────────────────────
# Signal 1: 100 Hz for the first second, then 300 Hz for the second second.
# The spectrogram should show a horizontal line that jumps at t=1.
x_switch = np.concatenate([
    np.cos(2 * np.pi * 100 * t[:f_s]),
    np.cos(2 * np.pi * 300 * t[:f_s]),
])
freqs2, times2, Sxx2 = stft(x_switch, f_s, nperseg=128, noverlap=96)
Sxx2_db = 10 * np.log10(Sxx2 + 1e-12)


# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Spectrogram — Short-Time Fourier Transform (STFT)\n'
             'X = time, Y = frequency, colour = power (dB)', fontsize=13)

# Chirp — time domain
axes[0, 0].plot(t, x_chirp, color='steelblue', linewidth=0.5)
axes[0, 0].set_title('Chirp: time domain  (50 → 400 Hz sweep)', fontsize=10)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# Chirp — spectrogram
im1 = axes[0, 1].pcolormesh(times, freqs, Sxx_db,
                              shading='gouraud', cmap='inferno',
                              vmin=-60, vmax=0)
axes[0, 1].set_title('Chirp: spectrogram  (diagonal line = rising freq)', fontsize=10)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Frequency (Hz)')
axes[0, 1].set_ylim([0, 450])
plt.colorbar(im1, ax=axes[0, 1], label='Power (dB)')

# Switching tones — time domain
t_full = np.linspace(0, 2.0, 2 * f_s, endpoint=False)
axes[1, 0].plot(t_full, x_switch, color='crimson', linewidth=0.5)
axes[1, 0].set_title('Switching tones: time domain  (100 Hz then 300 Hz)', fontsize=10)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)

# Switching tones — spectrogram
im2 = axes[1, 1].pcolormesh(times2, freqs2, Sxx2_db,
                              shading='gouraud', cmap='inferno',
                              vmin=-60, vmax=0)
axes[1, 1].set_title('Switching tones: spectrogram  (jump at t=1s)', fontsize=10)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Frequency (Hz)')
axes[1, 1].set_ylim([0, 450])
plt.colorbar(im2, ax=axes[1, 1], label='Power (dB)')

plt.tight_layout()
plt.savefig('spectrogram.png', dpi=150)
print("Plot saved → spectrogram.png")
plt.show()


# ── Window size tradeoff demo ─────────────────────────────────────────
print()
print("=== Time vs Frequency Resolution Tradeoff ===\n")
print(f"  {'nperseg':>10}  {'Δt (ms)':>10}  {'Δf (Hz)':>10}  {'Note'}")
print("  " + "-" * 52)
for nperseg in [32, 64, 128, 256, 512]:
    step  = nperseg // 4            # 75% overlap
    dt_ms = step / f_s * 1000
    df_hz = f_s / nperseg
    note  = ("fine time" if nperseg <= 64 else
             "fine freq" if nperseg >= 256 else "balanced")
    print(f"  {nperseg:>10}  {dt_ms:>10.1f}  {df_hz:>10.1f}  {note}")

print()
print("  5G NR OFDM relevance:")
print("  subcarrier spacing Δf = 15/30/60/120 kHz (numerology μ=0,1,2,3)")
print("  symbol duration   Δt = 1/Δf")
print("  Same tradeoff: wider subcarriers = shorter symbols = better for mobility.")
