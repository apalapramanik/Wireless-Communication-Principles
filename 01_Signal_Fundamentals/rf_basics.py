"""
Topic 2 — RF & Wireless Basics
Frequency / wavelength relationships and dB / dBm conversions.

Key formula:  c = f × λ   (c = 3×10⁸ m/s)
dBm = 10 × log10(P_watts / 0.001)

Run:
    python rf_basics.py
"""

import numpy as np

C = 3e8   # speed of light, m/s


# ── Frequency ↔ Wavelength ──────────────────────────────────────────

def wavelength(freq_hz: float) -> float:
    """λ = c / f  →  result in metres."""
    return C / freq_hz


def freq_to_band(freq_hz: float) -> str:
    if freq_hz < 3e9:
        return "Sub-3 GHz   (4G / WiFi 2.4 GHz territory)"
    elif freq_hz < 6e9:
        return "Sub-6 GHz   (5G NR FR1)"
    elif freq_hz < 30e9:
        return "cmWave      (5G NR FR1 upper / satellite)"
    else:
        return "mmWave      (5G NR FR2 — 28 / 39 GHz)"


# ── dB / dBm conversions ────────────────────────────────────────────

def watts_to_dbm(power_w: float) -> float:
    """P(dBm) = 10 × log10(P_W / 1e-3)"""
    return 10 * np.log10(power_w / 1e-3)


def dbm_to_watts(power_dbm: float) -> float:
    """P(W) = 1e-3 × 10^(P_dBm / 10)"""
    return 1e-3 * 10 ** (power_dbm / 10)


def power_ratio_db(p1_dbm: float, p2_dbm: float) -> str:
    """Express the ratio p1/p2 in both dB and as a linear multiplier."""
    diff   = p1_dbm - p2_dbm
    linear = 10 ** (diff / 10)
    return f"{diff:+.1f} dB  =  {linear:.1f}× power ratio"


# ── Shannon capacity (bonus — links BW to data rate) ────────────────

def shannon_capacity(bandwidth_hz: float, snr_linear: float) -> float:
    """C = B × log2(1 + SNR)  →  result in bits/sec."""
    return bandwidth_hz * np.log2(1 + snr_linear)


if __name__ == "__main__":
    # --- Frequency / wavelength table ---
    bands = {
        "700 MHz  (4G low band)  ": 700e6,
        "2.1 GHz  (4G mid band)  ": 2.1e9,
        "3.5 GHz  (5G NR FR1)    ": 3.5e9,
        "28 GHz   (5G mmWave)    ": 28e9,
        "39 GHz   (5G mmWave)    ": 39e9,
    }

    print("=== Frequency → Wavelength → Band ===\n")
    print(f"  {'Band':<30}  {'λ (cm)':>8}  {'Category'}")
    print("  " + "-" * 75)
    for name, f in bands.items():
        lam = wavelength(f)
        cat = freq_to_band(f)
        print(f"  {name}  {lam * 100:>8.2f}  {cat}")

    # --- Power conversion table ---
    print("\n=== Power Conversions ===\n")
    print(f"  {'Power (W)':>12}  {'dBm':>8}")
    print("  " + "-" * 25)
    for p in [10, 1, 0.1, 1e-3, 1e-6, 1e-9]:
        print(f"  {p:>12.6f}  {watts_to_dbm(p):>8.1f} dBm")

    # --- Ratio examples ---
    print("\n=== Power Ratios ===\n")
    examples = [
        (30,  0,   "gNB Tx (1 W) vs reference (1 mW)"),
        (-70, -100, "Typical UE receive vs sensitivity floor"),
        (0,   -3,   "3 dB loss = half the power"),
        (0,   -10,  "10 dB loss = one tenth the power"),
    ]
    for p1, p2, note in examples:
        print(f"  {p1:+4d} dBm vs {p2:+4d} dBm  →  {power_ratio_db(p1, p2)}   ({note})")

    # --- Shannon capacity ---
    print("\n=== Shannon Capacity: Bandwidth → Data Rate ===\n")
    scenarios = [
        ("4G  20 MHz   SNR=20 dB", 20e6,  100),
        ("5G  100 MHz  SNR=20 dB", 100e6, 100),
        ("5G  100 MHz  SNR=30 dB", 100e6, 1000),
        ("mmWave 400 MHz SNR=30dB", 400e6, 1000),
    ]
    for label, bw, snr_db in scenarios:
        snr_lin = 10 ** (snr_db / 10)
        cap = shannon_capacity(bw, snr_lin) / 1e6
        print(f"  {label:<28}  C = {cap:>8.1f} Mbps")

    print()
    print("  Takeaway: wide BW + good SNR = high capacity.")
    print("  Bandwidth sets the ceiling; SNR determines how close you get.")
