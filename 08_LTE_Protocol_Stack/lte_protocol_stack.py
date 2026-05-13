"""
Phase 2 | Topic 4: 4G LTE Protocol Stack
Simulations:
    1. HARQ with incremental redundancy combining
    2. RLC AM segmentation and reassembly
    3. MAC scheduling: Proportional Fair vs Maximum Throughput
    4. End-to-end sublayer latency breakdown
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque, defaultdict
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)


# ===========================================================================
# Helper: AWGN channel
# ===========================================================================

def awgn_channel(bits, snr_db):
    """
    Simulate BPSK over AWGN. Returns (decoded_bits, llrs).
    We work with Log Likelihood Ratios for soft combining in HARQ.
    """
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr_linear))
    bpsk = 2 * bits.astype(float) - 1          # 0 -> -1, 1 -> +1
    received = bpsk + rng.normal(0, sigma, len(bits))
    llrs = 2 * received / (sigma ** 2)          # LLR = 2r/sigma^2
    decoded = (llrs > 0).astype(int)
    return decoded, llrs


def compute_ber(bits_tx, bits_rx):
    return np.mean(bits_tx != bits_rx)


# ===========================================================================
# 1. HARQ: Hybrid Automatic Repeat Request with incremental redundancy
# ===========================================================================

def simulate_harq(snr_db, n_bits=256, max_retx=4, n_trials=500):
    """
    Simulate one HARQ process.
    Incremental Redundancy: each retransmission adds redundancy bits.
    The receiver soft-combines all received LLRs before decoding.

    Returns dict with:
        first_tx_success_rate
        avg_retx_count
        effective_throughput  (fraction of slots used for actual data)
        per_retx_ber          (BER after each combining round)
    """
    # Simple rate-1/3 turbo code approximation:
    # Transmission 0: systematic bits (rate 1)
    # Transmission 1: parity1 bits  (rate 1)
    # Transmission 2: parity2 bits  (rate 1)
    # Transmission 3: repeat systematic (rate 1)
    # Combined after k transmissions approaches rate 1/(k+1)

    results = {
        "success_round": [],       # which round succeeded (0 = first tx)
        "per_round_ber": [[] for _ in range(max_retx + 1)],
    }

    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, n_bits)

        # Generate redundancy versions (simplified: rotate bit pattern)
        rv_bits = [info_bits]
        for rv in range(1, max_retx):
            # In real LTE, different puncturing patterns are used.
            # Here we simulate by generating independent parity streams
            # that carry additional soft information.
            parity = (info_bits + rng.integers(0, 2, n_bits)) % 2
            rv_bits.append(parity)

        accumulated_llr = np.zeros(n_bits)
        success = False

        for retx in range(max_retx + 1):
            bits_to_tx = rv_bits[min(retx, len(rv_bits) - 1)]
            _, llr = awgn_channel(bits_to_tx, snr_db)

            # Chase combining: soft-combine LLRs
            if retx == 0:
                accumulated_llr = llr
            else:
                accumulated_llr = accumulated_llr + llr   # MRC combining

            decoded = (accumulated_llr > 0).astype(int)
            ber = compute_ber(info_bits, decoded)
            results["per_round_ber"][retx].append(ber)

            # CRC check approximation: success if BER is very low
            if ber == 0.0:
                results["success_round"].append(retx)
                success = True
                break

        if not success:
            results["success_round"].append(max_retx + 1)   # failed

    success_rounds = np.array(results["success_round"])
    first_tx_success = np.mean(success_rounds == 0)
    avg_retx = np.mean(success_rounds[success_rounds <= max_retx])
    # Throughput: 1 slot of info per (avg_retx + 1) slots used
    effective_tp = 1.0 / (avg_retx + 1) if avg_retx < max_retx + 1 else 0.0
    per_round_ber = [
        np.mean(results["per_round_ber"][r]) for r in range(max_retx + 1)
    ]

    return {
        "first_tx_success_rate": first_tx_success,
        "avg_retx_count": avg_retx,
        "effective_throughput": effective_tp,
        "per_round_ber": per_round_ber,
    }


def plot_harq():
    snr_range = np.arange(-6, 16, 2)
    first_tx_success = []
    eff_throughput = []
    per_round_ber_all = []

    print("Running HARQ simulation across SNR range...")
    for snr in snr_range:
        res = simulate_harq(snr_db=snr)
        first_tx_success.append(res["first_tx_success_rate"])
        eff_throughput.append(res["effective_throughput"])
        per_round_ber_all.append(res["per_round_ber"])

    # BER improvement across combining rounds at SNR = 2 dB
    demo_snr = 2
    demo_result = simulate_harq(snr_db=demo_snr, n_trials=1000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "HARQ with Incremental Redundancy (Soft Combining)",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: First transmission success rate
    axes[0].plot(snr_range, first_tx_success, "b-o", linewidth=2, markersize=6)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("First TX success rate")
    axes[0].set_title("First Transmission ACK Rate")
    axes[0].axhline(0.9, color="red", linestyle="--", alpha=0.6, label="90% target")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Panel 2: Effective throughput vs SNR
    axes[1].plot(snr_range, eff_throughput, "g-s", linewidth=2, markersize=6)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Effective throughput (fraction)")
    axes[1].set_title("Effective Throughput After HARQ")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Panel 3: BER reduction with combining rounds at demo_snr
    rounds = np.arange(len(demo_result["per_round_ber"]))
    ber_vals = demo_result["per_round_ber"]
    axes[2].semilogy(rounds, ber_vals, "r-D", linewidth=2, markersize=7)
    axes[2].set_xlabel("Transmission round (0 = first TX)")
    axes[2].set_ylabel("BER")
    axes[2].set_title(f"BER vs Combining Round (SNR = {demo_snr} dB)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(rounds)
    axes[2].set_xticklabels([f"TX{i}" for i in rounds])
    axes[2].annotate(
        "Soft combining\nimproves BER\nwith each retx",
        xy=(1, ber_vals[1]),
        xytext=(2.2, ber_vals[1] * 3),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig("harq_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: harq_simulation.png\n")


# ===========================================================================
# 2. RLC AM: Segmentation and Reassembly
# ===========================================================================

class RLC_AM_Tx:
    """
    RLC Acknowledged Mode Transmitter.
    Accepts SDUs (full IP packets), segments them into PDUs, tracks ACKs.
    """

    def __init__(self, max_pdu_bytes=128):
        self.max_pdu_bytes = max_pdu_bytes
        self.sdu_buffer = deque()      # incoming SDUs waiting to be segmented
        self.tx_pdu_buffer = {}        # sn -> PDU bytes (waiting for ACK)
        self.sn = 0                    # next sequence number to assign
        self.stats = {
            "sdus_received": 0,
            "pdus_sent": 0,
            "pdus_retransmitted": 0,
            "bytes_overhead": 0,       # RLC header bytes added
        }

    def receive_sdu(self, sdu_bytes):
        """Accept an SDU (IP packet) from PDCP."""
        self.sdu_buffer.append(sdu_bytes)
        self.stats["sdus_received"] += 1

    def _segment_next_sdu(self):
        """Segment the next SDU into PDUs and add to tx buffer."""
        if not self.sdu_buffer:
            return []
        sdu = self.sdu_buffer.popleft()
        pdus = []
        offset = 0
        while offset < len(sdu):
            payload = sdu[offset: offset + self.max_pdu_bytes]
            # RLC AM header: 2 bytes (SN + flags)
            pdu = {"sn": self.sn, "data": payload, "header_bytes": 2}
            self.tx_pdu_buffer[self.sn] = pdu
            pdus.append(pdu)
            self.sn += 1
            offset += self.max_pdu_bytes
            self.stats["pdus_sent"] += 1
            self.stats["bytes_overhead"] += 2
        return pdus

    def get_pdus_for_mac(self, n=4):
        """Return up to n PDUs ready to pass to MAC."""
        pdus = []
        while len(pdus) < n and self.sdu_buffer:
            pdus.extend(self._segment_next_sdu())
        return pdus[:n]

    def receive_ack(self, sn):
        """ACK from peer: remove PDU from retx buffer."""
        self.tx_pdu_buffer.pop(sn, None)

    def receive_nack(self, sn):
        """NACK from peer: mark PDU for retransmission."""
        if sn in self.tx_pdu_buffer:
            self.stats["pdus_retransmitted"] += 1
            return self.tx_pdu_buffer[sn]
        return None


class RLC_AM_Rx:
    """
    RLC Acknowledged Mode Receiver.
    Receives PDUs, handles reordering, reassembles SDUs.
    """

    def __init__(self):
        self.rx_buffer = {}           # sn -> PDU
        self.next_expected_sn = 0
        self.reassembled_sdus = []
        self.stats = {"pdus_received": 0, "out_of_order": 0, "sdus_delivered": 0}

    def receive_pdu(self, pdu):
        """Receive a PDU from the air interface."""
        sn = pdu["sn"]
        self.stats["pdus_received"] += 1
        if sn < self.next_expected_sn:
            return   # duplicate, discard
        self.rx_buffer[sn] = pdu
        if sn != self.next_expected_sn:
            self.stats["out_of_order"] += 1
        self._try_deliver()

    def _try_deliver(self):
        """Deliver in-order PDUs up the stack."""
        while self.next_expected_sn in self.rx_buffer:
            pdu = self.rx_buffer.pop(self.next_expected_sn)
            self.reassembled_sdus.append(pdu)
            self.stats["sdus_delivered"] += 1
            self.next_expected_sn += 1


def simulate_rlc():
    """
    Drive RLC TX and RX with random SDU sizes, random packet loss,
    and measure segmentation overhead and reorder events.
    """
    tx = RLC_AM_Tx(max_pdu_bytes=128)
    rx = RLC_AM_Rx()

    # Generate 20 random IP packets (200 to 1400 bytes)
    n_sdus = 20
    sdu_sizes = rng.integers(200, 1400, n_sdus)
    total_payload_bytes = 0

    for size in sdu_sizes:
        sdu = rng.integers(0, 256, size, dtype=np.uint8)
        tx.receive_sdu(sdu)
        total_payload_bytes += size

    # Transmit PDUs with 10% random loss
    loss_prob = 0.10
    all_pdus = tx.get_pdus_for_mac(n=10000)

    # Shuffle to simulate reordering (5% of PDUs delivered out of order)
    delivered = []
    for pdu in all_pdus:
        if rng.random() > loss_prob:
            delivered.append(pdu)

    # Deliver in random order to demonstrate reordering handling
    indices = list(range(len(delivered)))
    for i in range(0, len(indices) - 1, 10):
        if i + 1 < len(indices) and rng.random() < 0.05:
            indices[i], indices[i + 1] = indices[i + 1], indices[i]

    for idx in indices:
        rx.receive_pdu(delivered[idx])
        tx.receive_ack(delivered[idx]["sn"])

    # Count lost PDUs and retransmit
    lost_sns = set(range(tx.sn)) - set(delivered[i]["sn"] for i in indices)
    for sn in lost_sns:
        retx_pdu = tx.receive_nack(sn)
        if retx_pdu:
            rx.receive_pdu(retx_pdu)

    overhead_pct = tx.stats["bytes_overhead"] / total_payload_bytes * 100

    print("=" * 55)
    print("RLC AM Simulation Results")
    print("=" * 55)
    print(f"  SDUs submitted       : {tx.stats['sdus_received']}")
    print(f"  Total payload bytes  : {total_payload_bytes}")
    print(f"  PDUs generated       : {tx.stats['pdus_sent']}")
    print(f"  PDUs retransmitted   : {tx.stats['pdus_retransmitted']}")
    print(f"  RLC header overhead  : {tx.stats['bytes_overhead']} bytes ({overhead_pct:.1f}%)")
    print(f"  Out-of-order events  : {rx.stats['out_of_order']}")
    print(f"  PDUs delivered       : {rx.stats['sdus_delivered']}")
    print("=" * 55)

    return tx.stats, rx.stats, sdu_sizes, total_payload_bytes


# ===========================================================================
# 3. MAC Scheduler: Proportional Fair vs Maximum Throughput
# ===========================================================================

class UE:
    """Represents one UE in the scheduler simulation."""

    def __init__(self, ue_id, distance_m, n_rb_total):
        self.ue_id = ue_id
        self.distance_m = distance_m
        # Path loss determines average SNR: closer UE gets higher SNR
        # Using a simplified log-distance model: SNR decreases 3 dB per 2x distance
        ref_snr_db = 25.0   # SNR at 100 m
        path_loss_db = 35 * np.log10(distance_m / 100.0) if distance_m > 0 else 0
        self.mean_snr_db = ref_snr_db - path_loss_db
        self.n_rb = n_rb_total
        # Instantaneous SNR varies due to fading (Rayleigh, 6 dB std)
        self.throughput_history = deque(maxlen=100)   # for Proportional Fair averaging
        self.total_bytes = 0
        self.scheduled_count = 0

    def get_instantaneous_rate(self):
        """
        Instantaneous achievable rate (bits per resource block per subframe).
        Rayleigh fading adds random variation around mean SNR.
        """
        fading_db = 20 * np.log10(np.abs(rng.normal(1, 0.5)) + 0.01)
        snr_inst = self.mean_snr_db + fading_db
        snr_linear = 10 ** (snr_inst / 10)
        # Shannon rate per RB: 12 subcarriers * 14 symbols * log2(1+SNR) bits
        bits_per_rb = 12 * 14 * np.log2(1 + snr_linear)
        return max(bits_per_rb, 0)

    def get_avg_throughput(self):
        if not self.throughput_history:
            return 1.0   # avoid div by zero
        return np.mean(self.throughput_history)

    def update_throughput(self, bits_received):
        self.throughput_history.append(bits_received)
        self.total_bytes += bits_received / 8


def run_scheduler(n_ue=6, n_rb=50, n_subframes=1000, algorithm="proportional_fair"):
    """
    Simulate a MAC scheduler for n_subframes subframes.
    Each subframe, allocate n_rb resource blocks across UEs.

    Algorithms:
        'proportional_fair'   : maximise R_inst / R_avg
        'max_throughput'      : maximise R_inst
        'round_robin'         : equal turns
    """
    # UEs at varying distances: 100 m to 1500 m
    distances = np.linspace(100, 1500, n_ue)
    ues = [UE(i, d, n_rb) for i, d in enumerate(distances)]

    ue_bytes_per_subframe = defaultdict(list)

    for sf in range(n_subframes):
        # Each UE computes its instantaneous rate
        inst_rates = {ue.ue_id: ue.get_instantaneous_rate() for ue in ues}

        # Select UE to schedule based on algorithm
        if algorithm == "proportional_fair":
            pf_metrics = {
                ue.ue_id: inst_rates[ue.ue_id] / ue.get_avg_throughput()
                for ue in ues
            }
            scheduled_id = max(pf_metrics, key=pf_metrics.get)

        elif algorithm == "max_throughput":
            scheduled_id = max(inst_rates, key=inst_rates.get)

        elif algorithm == "round_robin":
            scheduled_id = sf % n_ue

        # Allocate all RBs to the scheduled UE (simplified single-user per SF)
        for ue in ues:
            bits = 0
            if ue.ue_id == scheduled_id:
                bits = inst_rates[ue.ue_id] * n_rb
                ue.scheduled_count += 1
                ue.total_bytes += bits / 8
            ue.update_throughput(bits)
            ue_bytes_per_subframe[ue.ue_id].append(bits / 8 / 1000)  # kB/subframe

    return ues, ue_bytes_per_subframe


def plot_scheduler():
    n_ue = 6
    distances = np.linspace(100, 1500, n_ue)

    print("Running scheduler simulations...")
    ues_pf, _ = run_scheduler(n_ue=n_ue, algorithm="proportional_fair")
    ues_mt, _ = run_scheduler(n_ue=n_ue, algorithm="max_throughput")
    ues_rr, _ = run_scheduler(n_ue=n_ue, algorithm="round_robin")

    def get_mbps(ues):
        return [ue.total_bytes / 1e6 for ue in ues]  # MB over 1000 subframes = ~1 Mb/s scale

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "MAC Scheduler Comparison: Proportional Fair vs Max Throughput vs Round Robin",
        fontsize=13, fontweight="bold"
    )
    x = np.arange(n_ue)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_ue))
    labels = [f"UE{i}\n({int(d)}m)" for i, d in enumerate(distances)]

    def make_bar_plot(ax, ues, title):
        vals = get_mbps(ues)
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("UE (distance from eNB)")
        ax.set_ylabel("Total data (MB) over 1000 subframes")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8
            )
        # Jain's fairness index
        n = len(vals)
        ji = (sum(vals) ** 2) / (n * sum(v ** 2 for v in vals)) if sum(vals) > 0 else 0
        ax.text(
            0.02, 0.97, f"Jain's fairness index: {ji:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7)
        )

    make_bar_plot(axes[0], ues_pf, "Proportional Fair")
    make_bar_plot(axes[1], ues_mt, "Maximum Throughput")
    make_bar_plot(axes[2], ues_rr, "Round Robin")

    plt.tight_layout()
    plt.savefig("mac_scheduler.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: mac_scheduler.png\n")


# ===========================================================================
# 4. Sublayer latency breakdown
# ===========================================================================

def plot_latency_breakdown():
    """
    Illustrate typical per-sublayer latency contributions in LTE downlink.
    Values based on 3GPP and industry estimates (ms).
    """
    layers = ["Core to eNB\n(transport)", "PDCP\n(header comp\n+ cipher)", "RLC\nsegment",
              "MAC\nscheduling", "PHY\nprocessing", "Air\ntransmit (1ms)", "UE PHY\nprocessing",
              "HARQ ACK\n(feedback)"]
    latencies = [1.5, 0.3, 0.2, 0.5, 0.5, 1.0, 0.5, 4.0]
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000",
              "#7030A0", "#00B0F0", "#7030A0", "#FF0000"]

    fig, ax = plt.subplots(figsize=(13, 5))

    bars = ax.bar(layers, latencies, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_ylabel("Latency contribution (ms)")
    ax.set_title(
        "LTE Downlink Latency Breakdown per Sublayer\n"
        "(Typical values, first-hop only, no retransmissions)",
        fontsize=12, fontweight="bold"
    )

    cumulative = 0
    for bar, val in zip(bars, latencies):
        cumulative += val
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.axhline(np.mean(latencies), color="grey", linestyle="--", alpha=0.5, label="Mean per sublayer")
    ax.text(
        len(layers) - 0.5, max(latencies) * 0.8,
        f"Total one-way: ~{sum(latencies):.1f} ms\n(LTE target: < 10 ms RTT)",
        ha="right", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(latencies) * 1.25)

    # Annotation: HARQ dominates
    ax.annotate(
        "HARQ round trip dominates\nwhen first TX fails",
        xy=(len(layers) - 1, latencies[-1]),
        xytext=(len(layers) - 3, latencies[-1] + 0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig("lte_latency_breakdown.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: lte_latency_breakdown.png\n")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\nPhase 2 | Topic 4: LTE Protocol Stack Simulation")
    print("=" * 55)

    print("\n[1/4] HARQ with incremental redundancy combining")
    plot_harq()

    print("[2/4] RLC AM segmentation and reassembly")
    simulate_rlc()

    print("\n[3/4] MAC scheduler comparison")
    plot_scheduler()

    print("[4/4] Sublayer latency breakdown")
    plot_latency_breakdown()

    print("\nAll simulations complete.")
    print("Output files: harq_simulation.png, mac_scheduler.png, lte_latency_breakdown.png")
