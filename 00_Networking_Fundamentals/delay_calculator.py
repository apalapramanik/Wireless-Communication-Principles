"""
Topic 1 — Networking Fundamentals
Nodal delay breakdown + traffic intensity vs queuing delay plot.

The 4 delay components at each router/switch:
    d_total = d_proc + d_queue + d_trans + d_prop

Run:
    python delay_calculator.py
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_delays(
    packet_size_bits: int,
    link_rate_bps: float,
    distance_m: float,
    prop_speed_mps: float = 2e8,   # ~speed of light in fiber
    proc_delay_s: float = 1e-6,    # 1 microsecond
    arrival_rate_pps: float = 1000,
) -> dict:
    """
    Compute the four components of nodal delay.

    Args:
        packet_size_bits : L  — packet length in bits
        link_rate_bps    : R  — link capacity in bits/sec
        distance_m       : d  — physical link length in metres
        prop_speed_mps   : s  — propagation speed (2e8 ≈ light in fiber)
        proc_delay_s     :     fixed processing delay
        arrival_rate_pps : a  — average packet arrival rate (packets/sec)

    Returns:
        dict with each delay component in milliseconds, total, and traffic intensity.
    """
    d_trans = packet_size_bits / link_rate_bps      # L / R
    d_prop  = distance_m / prop_speed_mps           # d / s
    d_proc  = proc_delay_s

    # M/D/1 queue approximation: d_q = (rho / (1 - rho)) * d_trans
    traffic_intensity = (packet_size_bits * arrival_rate_pps) / link_rate_bps
    if traffic_intensity >= 1.0:
        d_queue = float("inf")
    else:
        d_queue = (traffic_intensity / (1 - traffic_intensity)) * d_trans

    total = d_proc + d_queue + d_trans + d_prop

    return {
        "processing_ms":     d_proc  * 1e3,
        "queueing_ms":       d_queue * 1e3,
        "transmission_ms":   d_trans * 1e3,
        "propagation_ms":    d_prop  * 1e3,
        "total_ms":          total   * 1e3,
        "traffic_intensity": traffic_intensity,
    }


def plot_traffic_intensity():
    """
    Reproduce the classic Kurose & Ross Fig 1.18:
    queuing delay explodes as traffic intensity La/R → 1.
    """
    rho = np.linspace(0, 0.99, 500)
    norm_delay = rho / (1 - rho)   # normalised by d_trans

    plt.figure(figsize=(8, 5))
    plt.plot(rho, norm_delay, color="steelblue", linewidth=2)
    plt.axvline(x=1.0, color="red", linestyle="--", label="Saturation  La/R = 1")
    plt.xlabel("Traffic Intensity  La/R", fontsize=12)
    plt.ylabel("Normalised Queuing Delay  (d_q / d_trans)", fontsize=12)
    plt.title("Queuing Delay vs Traffic Intensity\n(M/D/1 model)", fontsize=13)
    plt.ylim(0, 20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("traffic_intensity.png", dpi=150)
    print("Plot saved → traffic_intensity.png")
    plt.show()


if __name__ == "__main__":
    # --- Example: 1500-byte Ethernet frame over a 1 Mbps link, 5 km away ---
    result = compute_delays(
        packet_size_bits=1500 * 8,
        link_rate_bps=1e6,
        distance_m=5_000,
        arrival_rate_pps=80,
    )

    print("=== Nodal Delay Breakdown ===")
    print(f"  {'Component':<22} {'Value':>12}")
    print("  " + "-" * 36)
    for k, v in result.items():
        unit = "" if k == "traffic_intensity" else " ms"
        print(f"  {k:<22} {v:>12.4f}{unit}")

    print()

    # --- Show how delay blows up near saturation ---
    print("=== Queuing Delay at Various Traffic Intensities ===")
    # Max sustainable rate for 1500-byte packets on 1 Mbps ≈ 83 pps
    for rate in [10, 30, 50, 65, 75, 80, 83]:
        r = compute_delays(
            packet_size_bits=1500 * 8,
            link_rate_bps=1e6,
            distance_m=5_000,
            arrival_rate_pps=rate,
        )
        q = r["queueing_ms"]
        ti = r["traffic_intensity"]
        q_str = f"{q:.2f} ms" if q != float("inf") else "∞  (queue saturated!)"
        print(f"  arrival={rate:4d} pps  La/R={ti:.3f}  →  queueing delay = {q_str}")

    print()
    plot_traffic_intensity()
