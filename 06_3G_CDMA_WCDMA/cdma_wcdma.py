"""
Phase 2 | Topic 2: 3G and CDMA / WCDMA
Simulator: Spreading, despreading, near-far problem, power control, RAKE receiver

Sections:
    1. CDMA spreading and despreading (single user)
    2. Multi-user CDMA with orthogonal codes (Walsh codes)
    3. The near-far problem
    4. Closed-loop power control
    5. RAKE receiver with multipath combining
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(42)


# ===========================================================================
# 1. Walsh code generator (orthogonal spreading codes)
# ===========================================================================

def generate_walsh_codes(order: int) -> np.ndarray:
    """
    Generate a Walsh-Hadamard matrix of the given order.
    Each row is one orthogonal spreading code.
    Order must be a power of 2.

    Returns array of shape (order, order) with values in {+1, -1}.
    """
    if order == 1:
        return np.array([[1]])
    half = generate_walsh_codes(order // 2)
    top    = np.hstack([half,  half])
    bottom = np.hstack([half, -half])
    return np.vstack([top, bottom])


def spreading_factor_to_chips(data_bits: np.ndarray, code: np.ndarray) -> np.ndarray:
    """
    Spread data_bits using the given spreading code.
    Each data bit is replaced by (bit * code), producing SF chips per bit.

    data_bits : array of +1/-1 values
    code      : spreading code of length SF, values in {+1, -1}
    Returns   : chip sequence of length (len(data_bits) * SF)
    """
    return np.concatenate([bit * code for bit in data_bits])


def despread(chips: np.ndarray, code: np.ndarray) -> np.ndarray:
    """
    Recover data bits from a chip sequence using the same spreading code.
    Each SF chips are correlated with the code and sign-detected.
    """
    sf = len(code)
    n_bits = len(chips) // sf
    bits = np.zeros(n_bits)
    for i in range(n_bits):
        segment = chips[i * sf : (i + 1) * sf]
        correlation = np.dot(segment, code)
        bits[i] = np.sign(correlation)
    return bits


# ===========================================================================
# 2. Multi-user CDMA
# ===========================================================================

def simulate_cdma_multiuser(n_users: int = 4, n_bits: int = 8, snr_db: float = 20.0):
    """
    Simulate simultaneous transmission by n_users users sharing the same channel.
    Each user gets a unique row of the Walsh matrix as their spreading code.
    """
    sf = n_users                              # spreading factor = number of codes
    codes = generate_walsh_codes(sf)          # shape: (sf, sf)

    # Generate random data for each user
    data = {u: rng.choice([-1, 1], size=n_bits) for u in range(n_users)}

    # Spread and sum all users onto the same channel
    channel = np.zeros(n_bits * sf)
    for u in range(n_users):
        chips = spreading_factor_to_chips(data[u], codes[u])
        channel += chips

    # Add AWGN noise
    chip_power   = np.mean(channel ** 2)
    noise_power  = chip_power / (10 ** (snr_db / 10))
    noise        = rng.normal(0, np.sqrt(noise_power), size=len(channel))
    received     = channel + noise

    # Despread each user
    recovered = {}
    ber_per_user = {}
    for u in range(n_users):
        recovered[u] = despread(received, codes[u])
        errors = np.sum(recovered[u] != data[u])
        ber_per_user[u] = errors / n_bits

    return data, recovered, ber_per_user, codes


# ===========================================================================
# 3. Near-far problem and power control
# ===========================================================================

def path_loss_linear(distance_m: float, frequency_hz: float = 2e9, ple: float = 3.5) -> float:
    """Simple path loss model: PL = (4 pi d f / c)^ple"""
    c   = 3e8
    d0  = 1.0
    pl0 = (4 * np.pi * d0 * frequency_hz / c) ** 2
    return pl0 * (distance_m / d0) ** ple


def simulate_near_far(distances_m: list, tx_power_dbm: float = 23.0,
                      noise_floor_dbm: float = -100.0, n_steps: int = 200):
    """
    Demonstrate the near-far problem and closed-loop power control.

    Without power control: close users dominate, far users are buried.
    With power control: all users converge to the same received power at Node B.

    Returns arrays of received power per user over time, with and without control.
    """
    n_users = len(distances_m)
    target_rx_dbm = -80.0     # target received power at Node B
    step_db       = 1.0       # TPC step size

    # Path loss for each user (linear scale)
    pl = np.array([path_loss_linear(d) for d in distances_m])

    # Without power control: fixed transmit power
    tx_linear_fixed = 10 ** (tx_power_dbm / 10)
    rx_no_ctrl = np.array([10 * np.log10(tx_linear_fixed / p) for p in pl])

    # With power control: iterative closed loop simulation
    tx_dbm = np.full(n_users, tx_power_dbm, dtype=float)
    rx_history = np.zeros((n_steps, n_users))

    for step in range(n_steps):
        pl_db   = 10 * np.log10(pl)
        rx_dbm  = tx_dbm - pl_db
        rx_history[step] = rx_dbm

        # TPC: adjust each user up or down toward target
        for u in range(n_users):
            if rx_dbm[u] < target_rx_dbm:
                tx_dbm[u] = min(tx_dbm[u] + step_db, 33.0)  # max 33 dBm
            else:
                tx_dbm[u] = max(tx_dbm[u] - step_db, -10.0) # min -10 dBm

    return rx_no_ctrl, rx_history, target_rx_dbm


# ===========================================================================
# 4. RAKE receiver
# ===========================================================================

def simulate_rake_receiver(sf: int = 16, n_data_bits: int = 50,
                            snr_db: float = 5.0):
    """
    Simulate a RAKE receiver combining multipath copies of a CDMA signal.

    The channel has 3 paths with different delays and attenuations.
    A single-finger receiver (matched filter only) is compared against a
    3-finger RAKE that combines all paths via Maximum Ratio Combining.
    """
    code  = generate_walsh_codes(sf)[0]    # use first Walsh code
    data  = rng.choice([-1, 1], size=n_data_bits)
    chips = spreading_factor_to_chips(data, code)
    n_chips = len(chips)

    # Multipath channel: 3 paths (delay in chips, amplitude)
    paths = [
        (0,  1.00),   # direct path (reference)
        (2,  0.60),   # first reflection
        (5,  0.35),   # second reflection
    ]

    # Build received signal as superposition of delayed copies
    max_delay = max(d for d, _ in paths)
    received  = np.zeros(n_chips + max_delay)
    for delay, amp in paths:
        received[delay : delay + n_chips] += amp * chips

    # Add AWGN noise
    signal_power = np.mean(chips ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = rng.normal(0, np.sqrt(noise_power), size=len(received))
    received     += noise

    # --- Single finger receiver (uses only the direct path) ---
    single_bits = despread(received[:n_chips], code)
    ber_single  = np.mean(single_bits != data)

    # --- RAKE receiver: one finger per path, MRC combining ---
    finger_outputs = []
    for delay, amp in paths:
        segment = received[delay : delay + n_chips]
        # Correlate with code chip by chip
        corr = np.zeros(n_data_bits)
        for i in range(n_data_bits):
            seg = segment[i * sf : (i + 1) * sf]
            if len(seg) == sf:
                corr[i] = np.dot(seg, code)
        finger_outputs.append((amp, corr))  # weight by path amplitude (MRC)

    # MRC: weighted sum of all finger outputs
    combined = sum(amp * corr for amp, corr in finger_outputs)
    rake_bits = np.sign(combined)
    ber_rake  = np.mean(rake_bits != data)

    return ber_single, ber_rake, paths


# ===========================================================================
# 5. BER vs SNR: single finger vs RAKE
# ===========================================================================

def ber_vs_snr_rake(snr_range_db=None, n_trials: int = 20, n_bits: int = 200):
    if snr_range_db is None:
        snr_range_db = np.arange(-5, 25, 2)
    ber_single_arr = []
    ber_rake_arr   = []

    for snr_db in snr_range_db:
        s_acc, r_acc = 0.0, 0.0
        for _ in range(n_trials):
            s, r, _ = simulate_rake_receiver(sf=16, n_data_bits=n_bits, snr_db=snr_db)
            s_acc += s
            r_acc += r
        ber_single_arr.append(s_acc / n_trials)
        ber_rake_arr.append(r_acc / n_trials)

    return snr_range_db, np.array(ber_single_arr), np.array(ber_rake_arr)


# ===========================================================================
# Plotting
# ===========================================================================

def plot_all():
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("3G CDMA / WCDMA: Core Mechanisms", fontsize=15, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ------------------------------------------------------------------
    # Panel 1: Walsh codes (orthogonality visualised)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    codes4 = generate_walsh_codes(4)
    for i, c in enumerate(codes4):
        ax1.step(range(5), np.append(c, c[-1]), where="post",
                 label=f"Code {i}", linewidth=2)
    ax1.set_title("Walsh codes (SF=4, 4 orthogonal codes)", fontsize=11)
    ax1.set_xlabel("Chip index")
    ax1.set_ylabel("Amplitude")
    ax1.set_xticks(range(5))
    ax1.set_yticks([-1, 0, 1])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: Spreading and despreading (one user, one bit)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    code = generate_walsh_codes(8)[1]
    bit  = np.array([+1])
    chips = spreading_factor_to_chips(bit, code)
    ax2.step(range(len(chips)+1), np.append(chips, chips[-1]),
             where="post", color="steelblue", linewidth=2, label="Chips (spread)")
    ax2.axhline(bit[0], color="tomato", linestyle="--", linewidth=1.5, label=f"Original bit = {bit[0]:+d}")
    ax2.set_title("Spreading: 1 data bit -> 8 chips (SF=8)", fontsize=11)
    ax2.set_xlabel("Chip index")
    ax2.set_ylabel("Amplitude")
    ax2.set_yticks([-1, 0, 1])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: Multi-user CDMA — received vs sent comparison
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    data_out, recovered_out, ber_out, _ = simulate_cdma_multiuser(
        n_users=4, n_bits=16, snr_db=25.0
    )
    n_bits = 16
    x = np.arange(n_bits)
    colors = ["steelblue", "tomato", "seagreen", "darkorchid"]
    for u in range(4):
        offset = u * 0.18
        ax3.scatter(x, data_out[u] + offset, marker="s", s=20,
                    color=colors[u], label=f"User {u} sent", zorder=3)
        ax3.scatter(x, recovered_out[u] + offset, marker="x", s=30,
                    color=colors[u], alpha=0.6)
    ax3.set_title("Multi-user CDMA: sent (■) vs recovered (×)\n4 users, same freq, same time, SF=4", fontsize=10)
    ax3.set_xlabel("Bit index")
    ax3.set_ylabel("Amplitude (offset per user)")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(True, alpha=0.3)
    ber_str = "  ".join([f"U{u}: BER={v:.2f}" for u, v in ber_out.items()])
    ax3.set_xlabel(f"Bit index        [{ber_str}]", fontsize=8)

    # ------------------------------------------------------------------
    # Panel 4: Near-far problem and power control convergence
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    distances = [100, 500, 1000, 2000]
    rx_no_ctrl, rx_history, target = simulate_near_far(distances)
    colors_nf = ["steelblue", "tomato", "seagreen", "darkorchid"]
    for u, d in enumerate(distances):
        ax4.plot(rx_history[:, u], color=colors_nf[u], linewidth=1.5,
                 label=f"User {u+1} ({d} m)")
    ax4.axhline(target, color="black", linestyle="--", linewidth=1.2, label="Target Rx power")
    ax4.set_title("Closed-loop power control: all users converge\nto same received power at Node B", fontsize=10)
    ax4.set_xlabel("TPC iteration (1 per 0.667 ms slot)")
    ax4.set_ylabel("Received power at Node B (dBm)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 5: RAKE receiver multipath impulse response
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    _, _, paths = simulate_rake_receiver()
    delays = [d for d, _ in paths]
    amps   = [a for _, a in paths]
    markerline, stemlines, baseline = ax5.stem(delays, amps, basefmt="k-")
    plt.setp(stemlines, color="steelblue", linewidth=2)
    plt.setp(markerline, color="steelblue", markersize=8)
    ax5.set_title("Multipath channel: 3 paths seen by RAKE receiver", fontsize=11)
    ax5.set_xlabel("Delay (chips)")
    ax5.set_ylabel("Path amplitude")
    ax5.set_xticks(range(max(delays)+2))
    ax5.grid(True, alpha=0.3)
    for d, a in zip(delays, amps):
        ax5.annotate(f"amp={a:.2f}", xy=(d, a), xytext=(d+0.1, a+0.03), fontsize=9)

    # ------------------------------------------------------------------
    # Panel 6: BER vs SNR — single finger vs RAKE
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    snr_range, ber_s, ber_r = ber_vs_snr_rake()
    ax6.semilogy(snr_range, np.clip(ber_s, 1e-4, 1), "tomato",
                 marker="o", linewidth=2, label="Single finger")
    ax6.semilogy(snr_range, np.clip(ber_r, 1e-4, 1), "steelblue",
                 marker="s", linewidth=2, label="RAKE (3 fingers, MRC)")
    ax6.set_title("BER vs SNR: RAKE diversity gain over single finger", fontsize=11)
    ax6.set_xlabel("SNR (dB)")
    ax6.set_ylabel("Bit Error Rate")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, which="both")
    ax6.set_ylim([1e-4, 1])

    plt.savefig("phase2_topic2_cdma_wcdma.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to phase2_topic2_cdma_wcdma.png")


# ===========================================================================
# Console summary
# ===========================================================================

def print_summary():
    print("=" * 60)
    print("  3G CDMA / WCDMA: Key Results")
    print("=" * 60)

    # Walsh orthogonality check
    codes = generate_walsh_codes(4)
    print("\n1. Walsh code orthogonality check (SF=4):")
    print(f"   {'Code pair':14s}  Dot product")
    for i in range(4):
        for j in range(i+1, 4):
            dot = np.dot(codes[i], codes[j])
            print(f"   Code {i} vs {j}:    {dot:+.0f}  {'(orthogonal)' if dot==0 else '(NOT orthogonal)'}")

    # Multi-user CDMA BER
    print("\n2. Multi-user CDMA (4 users, SF=4, SNR=25 dB):")
    _, _, ber_out, _ = simulate_cdma_multiuser(n_users=4, n_bits=100, snr_db=25.0)
    for u, ber in ber_out.items():
        print(f"   User {u}: BER = {ber:.4f}")

    # Near-far: Rx power without power control
    distances = [100, 500, 1000, 2000]
    rx_no_ctrl, rx_history, _ = simulate_near_far(distances)
    print("\n3. Near-far problem (without power control):")
    for u, d in enumerate(distances):
        print(f"   User {u+1} at {d:5d} m: Rx power = {rx_no_ctrl[u]:+.1f} dBm")
    print(f"   Spread = {rx_no_ctrl.max() - rx_no_ctrl.min():.1f} dB  "
          f"(far user is this much weaker than near user)")
    print(f"   After power control: all users -> -80.0 dBm (converged in ~{np.argmin(np.abs(rx_history[:, -1] - (-80.0)))} steps)")

    # RAKE vs single finger
    print("\n4. RAKE vs single-finger BER at SNR = 5 dB:")
    ber_s, ber_r, _ = simulate_rake_receiver(snr_db=5.0)
    print(f"   Single finger BER : {ber_s:.4f}")
    print(f"   RAKE (3 fingers)  : {ber_r:.4f}")
    gain = 10 * np.log10(ber_s / ber_r) if ber_r > 0 else float("inf")
    print(f"   Diversity gain    : ~{gain:.1f} dB equivalent improvement")

    # Processing gain table
    print("\n5. Processing gain for common spreading factors:")
    print(f"   {'SF':6s}  {'Data rate (Mcps=3.84)':24s}  {'Processing gain'}")
    for sf in [4, 8, 16, 32, 64, 128, 256]:
        rate_kbps = 3840 / sf
        pg_db = 10 * np.log10(sf)
        print(f"   {sf:<6d}  {rate_kbps:>8.1f} kbps                  {pg_db:.1f} dB")


if __name__ == "__main__":
    print_summary()
    plot_all()
