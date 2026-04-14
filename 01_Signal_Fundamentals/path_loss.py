"""
Topic 2 — RF & Wireless Basics
Path loss models + log-normal shadowing simulation.

Models covered:
  1. Free Space Path Loss (FSPL)        — ideal, LOS, no obstacles
  2. Log-Distance Path Loss             — general; n = path loss exponent
  3. Log-Normal Shadowing               — random variation on top of (2)

Key formulas:
  FSPL(dB) = 20·log10(d) + 20·log10(f) - 147.55
  PL(d)    = PL(d0) + 10·n·log10(d/d0) + Xσ
  Xσ ~ N(0, σ²)   in dB

Run:
    python path_loss.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Path loss models ────────────────────────────────────────────────

def fspl_db(distance_m: float | np.ndarray, freq_hz: float) -> float | np.ndarray:
    """
    Free Space Path Loss in dB.
    Assumes perfect LOS, no reflections, no atmosphere.
    """
    return 20 * np.log10(distance_m) + 20 * np.log10(freq_hz) - 147.55


def log_distance_pl(
    d: float | np.ndarray,
    d0: float,
    pl_d0_db: float,
    n: float,
) -> float | np.ndarray:
    """
    Log-distance path loss model.

    PL(d) = PL(d0) + 10·n·log10(d/d0)

    Args:
        d        : distance(s) in metres
        d0       : reference distance (usually 1 m or 100 m)
        pl_d0_db : measured path loss at d0 (dB)
        n        : path loss exponent (2 = free space, up to 6 indoors)
    """
    return pl_d0_db + 10 * n * np.log10(d / d0)


def simulate_shadowing(
    distances: np.ndarray,
    freq_hz: float = 3.5e9,
    n: float = 3.5,
    sigma_db: float = 8.0,
    d0: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add zero-mean Gaussian (log-normal) shadowing to the mean path loss.

    Returns:
        pl_mean     : deterministic log-distance path loss (dB)
        pl_shadowed : mean + random shadowing realisation (dB)
    """
    rng    = np.random.default_rng(seed)
    pl_d0  = fspl_db(d0, freq_hz)
    pl_mean = log_distance_pl(distances, d0, pl_d0, n)
    shadow  = rng.normal(0, sigma_db, size=len(distances))
    return pl_mean, pl_mean + shadow


# ── Plots ────────────────────────────────────────────────────────────

def plot_fspl_vs_frequency():
    """Compare FSPL for 700 MHz, 3.5 GHz, and 28 GHz — same distance axis."""
    distances = np.logspace(1, 4, 500)   # 10 m → 10 km

    freqs = {
        "700 MHz  (4G low band)":  (700e6,  "royalblue"),
        "3.5 GHz  (5G FR1)":       (3.5e9,  "forestgreen"),
        "28 GHz   (5G mmWave)":    (28e9,   "crimson"),
    }

    plt.figure(figsize=(10, 6))
    for label, (f, color) in freqs.items():
        pl = fspl_db(distances, f)
        plt.plot(distances / 1e3, pl, label=label, color=color, linewidth=2)

    plt.axhline(y=140, color="gray", linestyle="--", alpha=0.7,
                label="Typical max link budget ≈ 140 dB")
    plt.xscale("log")
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Free Space Path Loss (dB)", fontsize=12)
    plt.title("FSPL vs Distance — 4G vs 5G FR1 vs 5G mmWave\n"
              "(Higher freq → more loss → shorter range)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fspl_vs_frequency.png", dpi=150)
    print("Plot saved → fspl_vs_frequency.png")
    plt.show()


def plot_shadowing():
    """Plot mean path loss + a shadowed realisation at 3.5 GHz, urban NLOS."""
    d = np.linspace(10, 2_000, 1_000)
    pl_mean, pl_shadow = simulate_shadowing(d, freq_hz=3.5e9, n=3.5, sigma_db=8)

    plt.figure(figsize=(10, 5))
    plt.plot(d, pl_shadow, color="steelblue", alpha=0.6, linewidth=1,
             label="With log-normal shadowing  σ = 8 dB")
    plt.plot(d, pl_mean,   color="red",       linewidth=2,
             label="Mean path loss  n = 3.5 (urban NLOS)")

    plt.xlabel("Distance (m)", fontsize=12)
    plt.ylabel("Path Loss (dB)", fontsize=12)
    plt.title("Log-Distance Path Loss + Log-Normal Shadowing  @ 3.5 GHz\n"
              "(Why your signal fluctuates even when standing still)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("path_loss_shadowing.png", dpi=150)
    print("Plot saved → path_loss_shadowing.png")
    plt.show()


def ple_environment_table():
    """Print typical PLE values from 3GPP TR 38.901."""
    envs = {
        "Free space (any frequency)":      2.0,
        "Urban LOS — Sub-6 GHz":           2.2,
        "Urban LOS — mmWave 28 GHz":       2.1,
        "Urban NLOS — Sub-6 GHz":          3.5,
        "Urban NLOS — mmWave 28 GHz":      4.0,
        "Indoor — Sub-6 GHz":              3.0,
        "Indoor — mmWave 28 GHz":          5.5,
    }
    print("=== Path Loss Exponent (PLE) by Environment  [3GPP TR 38.901] ===\n")
    print(f"  {'Environment':<42} {'n':>5}")
    print("  " + "-" * 50)
    for env, n in envs.items():
        print(f"  {env:<42} {n:>5.1f}")
    print()
    print("  Key insight: LOS n ≈ 2 regardless of frequency.")
    print("  NLOS mmWave n is higher because 28 GHz cannot diffract around obstacles.")


if __name__ == "__main__":
    # Numeric sanity check
    print("=== FSPL spot checks ===\n")
    checks = [
        (1_000, 700e6,  "1 km, 700 MHz (4G)"),
        (1_000, 3.5e9,  "1 km, 3.5 GHz (5G FR1)"),
        (1_000, 28e9,   "1 km, 28 GHz (mmWave)"),
        (500,   3.5e9,  "500 m, 3.5 GHz"),
        (100,   28e9,   "100 m, 28 GHz"),
    ]
    for d, f, label in checks:
        pl = fspl_db(d, f)
        print(f"  {label:<30}  FSPL = {pl:.1f} dB")

    print()
    ple_environment_table()
    print()

    plot_fspl_vs_frequency()
    plot_shadowing()
