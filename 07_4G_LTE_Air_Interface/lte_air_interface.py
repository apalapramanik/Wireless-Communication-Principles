"""
Phase 2 | Topic 3: 4G LTE Air Interface
Simulator: Resource grid, OFDMA scheduling, SC-FDMA, link adaptation, throughput

Sections:
    1. LTE resource grid construction and visualisation
    2. OFDMA multi-user subframe allocation
    3. SC-FDMA uplink: DFT precoding vs plain OFDM (PAPR comparison)
    4. Link adaptation: MCS selection vs SNR
    5. Peak throughput calculator
    6. End-to-end OFDMA transceiver with channel estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(42)


# ===========================================================================
# LTE parameters
# ===========================================================================

LTE_PARAMS = {
    "subcarrier_spacing_hz": 15_000,
    "useful_symbol_duration_us": 1 / 15_000 * 1e6,   # 66.7 us
    "cp_normal_us": 4.7,
    "symbol_duration_us": 66.7 + 4.7,                 # 71.4 us
    "symbols_per_slot": 7,
    "slots_per_subframe": 2,
    "subframe_duration_ms": 1.0,
    "subcarriers_per_rb": 12,
    "re_per_rb_slot": 84,
}

BANDWIDTH_CONFIG = {
    # bandwidth_mhz: (n_rb, n_fft)
    1.4:  (6,   128),
    3:    (15,  256),
    5:    (25,  512),
    10:   (50,  1024),
    15:   (75,  1536),
    20:   (100, 2048),
}

# MCS table: (modulation_order, code_rate, label)
MCS_TABLE = [
    (2, 0.12, "QPSK r=1/8"),
    (2, 0.19, "QPSK r=1/5"),
    (2, 0.30, "QPSK r=1/3"),
    (2, 0.44, "QPSK r=1/2"),
    (2, 0.59, "QPSK r=2/3"),
    (2, 0.74, "QPSK r=3/4"),
    (4, 0.33, "16QAM r=1/3"),
    (4, 0.44, "16QAM r=1/2"),
    (4, 0.59, "16QAM r=2/3"),
    (4, 0.74, "16QAM r=3/4"),
    (4, 0.85, "16QAM r=5/6"),
    (6, 0.44, "64QAM r=1/2"),
    (6, 0.55, "64QAM r=5/9"),
    (6, 0.65, "64QAM r=2/3"),
    (6, 0.75, "64QAM r=3/4"),
    (6, 0.85, "64QAM r=5/6"),
    (6, 0.93, "64QAM r=15/16"),
]

# Approximate minimum SNR (dB) required for each MCS (10% BLER target)
MCS_SNR_THRESHOLD = [
    -6, -4, -2, 0, 2, 4,        # QPSK entries
    7, 9, 11, 13, 15,            # 16QAM entries
    18, 20, 22, 24, 26, 28,      # 64QAM entries
]


# ===========================================================================
# 1. LTE resource grid
# ===========================================================================

class LTEResourceGrid:
    """
    Represents one LTE subframe (1 ms) for a given bandwidth.
    RE types: DATA, PILOT, CONTROL, GUARD, DC, UNUSED
    """

    RE_TYPES = {
        "DATA":    0,
        "PILOT":   1,
        "CONTROL": 2,
        "GUARD":   3,
        "DC":      4,
    }

    def __init__(self, bandwidth_mhz: float = 10.0, n_antenna_ports: int = 1):
        cfg = BANDWIDTH_CONFIG[bandwidth_mhz]
        self.n_rb        = cfg[0]
        self.n_fft       = cfg[1]
        self.bw_mhz      = bandwidth_mhz
        self.n_sc        = self.n_rb * 12          # active subcarriers
        self.n_symbols   = 14                       # symbols per subframe (2 slots × 7)
        self.n_ports     = n_antenna_ports

        # Grid shape: (subcarriers, symbols)
        self.grid = np.zeros((self.n_sc, self.n_symbols), dtype=int)
        self._populate_control()
        self._populate_pilots()

    def _populate_control(self):
        """Mark first 3 symbols as PDCCH (control region)."""
        self.grid[:, :3] = self.RE_TYPES["CONTROL"]

    def _populate_pilots(self):
        """
        Place cell-specific reference signals (CRS) at standard positions.
        Pattern: symbol 0 and 4 in each slot, every 6th subcarrier, offset 0 or 3.
        """
        pilot_symbols = [0, 4, 7, 11]  # within the subframe (0-indexed)
        offsets       = [0, 3, 0, 3]

        for sym, off in zip(pilot_symbols, offsets):
            for sc in range(off, self.n_sc, 6):
                if self.grid[sc, sym] != self.RE_TYPES["CONTROL"]:
                    self.grid[sc, sym] = self.RE_TYPES["PILOT"]

    def allocate_rb_pair(self, rb_index: int, user_id: int):
        """Mark one RB pair (all 14 symbols, 12 subcarriers) as belonging to user_id."""
        sc_start = rb_index * 12
        sc_end   = sc_start + 12
        for sc in range(sc_start, sc_end):
            for sym in range(self.n_symbols):
                if self.grid[sc, sym] == self.RE_TYPES["DATA"]:
                    self.grid[sc, sym] = 10 + user_id  # user colour offset

    def count_data_res(self) -> int:
        return int(np.sum(self.grid == self.RE_TYPES["DATA"]))

    def overhead_fraction(self) -> float:
        total = self.n_sc * self.n_symbols
        pilot   = np.sum(self.grid == self.RE_TYPES["PILOT"])
        control = np.sum(self.grid == self.RE_TYPES["CONTROL"])
        return (pilot + control) / total


# ===========================================================================
# 2. OFDMA scheduler (proportional fair, simplified)
# ===========================================================================

def simulate_ofdma_scheduler(n_users: int = 5, n_rb: int = 25,
                               snr_range_db: tuple = (-5, 30)):
    """
    Simulate one subframe of proportional-fair scheduling over n_rb RBs.
    Each user has a random per-RB SNR. The scheduler assigns each RB to
    the user with the highest instantaneous rate / average rate ratio.
    Returns per-user RB assignments and achievable bits.
    """
    # Random per-user, per-RB SNR (models frequency-selective fading)
    snr_db = rng.uniform(snr_range_db[0], snr_range_db[1],
                         size=(n_users, n_rb))

    # Achievable bits per RE for each user/RB (Shannon bound, clipped to MCS range)
    bits_per_re = np.log2(1 + 10 ** (snr_db / 10))
    bits_per_re = np.clip(bits_per_re, 0, 6)  # max 6 bits/RE (64-QAM)

    # Average rate per user (initialised to mean across RBs)
    avg_rate = np.mean(bits_per_re, axis=1) + 1e-6

    rb_assignments = np.full(n_rb, -1, dtype=int)
    bits_allocated = np.zeros(n_users)

    for rb in range(n_rb):
        # Proportional fair metric: instantaneous / average
        pf_metric = bits_per_re[:, rb] / avg_rate
        winner = int(np.argmax(pf_metric))
        rb_assignments[rb] = winner
        bits_allocated[winner] += bits_per_re[winner, rb] * 84  # 84 RE per RB

    return rb_assignments, bits_allocated, snr_db, bits_per_re


# ===========================================================================
# 3. SC-FDMA vs OFDMA: PAPR comparison
# ===========================================================================

def compute_papr_db(signal: np.ndarray) -> float:
    peak   = np.max(np.abs(signal) ** 2)
    mean   = np.mean(np.abs(signal) ** 2)
    return 10 * np.log10(peak / mean)


def scfdma_vs_ofdma_papr(n_subcarriers: int = 12, n_symbols: int = 1000,
                          n_fft: int = 128):
    """
    Compare PAPR distributions for OFDMA vs SC-FDMA.
    SC-FDMA applies a DFT before mapping to subcarriers (DFT-s-OFDM).
    """
    papr_ofdma  = []
    papr_scfdma = []

    # Subcarrier indices (consecutive block allocation)
    sc_indices = np.arange(10, 10 + n_subcarriers)

    for _ in range(n_symbols):
        # Random QPSK data symbols
        data = rng.choice([-1-1j, -1+1j, 1-1j, 1+1j], size=n_subcarriers)

        # -- OFDMA: map directly to subcarriers, IFFT --
        freq_ofdma = np.zeros(n_fft, dtype=complex)
        freq_ofdma[sc_indices] = data
        time_ofdma = np.fft.ifft(freq_ofdma) * n_fft
        papr_ofdma.append(compute_papr_db(time_ofdma))

        # -- SC-FDMA: DFT precoding first, then map, then IFFT --
        dft_data = np.fft.fft(data)  # M-point DFT (M = n_subcarriers)
        freq_scfdma = np.zeros(n_fft, dtype=complex)
        freq_scfdma[sc_indices] = dft_data
        time_scfdma = np.fft.ifft(freq_scfdma) * n_fft
        papr_scfdma.append(compute_papr_db(time_scfdma))

    return np.array(papr_ofdma), np.array(papr_scfdma)


# ===========================================================================
# 4. Link adaptation: MCS selection and throughput vs SNR
# ===========================================================================

def select_mcs(snr_db: float) -> int:
    """Return the highest MCS index usable at the given SNR."""
    selected = 0
    for idx, threshold in enumerate(MCS_SNR_THRESHOLD):
        if snr_db >= threshold:
            selected = idx
    return selected


def throughput_vs_snr(n_rb: int = 50, snr_range=None):
    """
    Calculate achievable LTE downlink throughput vs SNR for a given bandwidth.
    Uses MCS table to select modulation and coding rate.
    RE count: n_rb × 84 RE/RB × 2 slots/subframe, minus overhead.
    """
    if snr_range is None:
        snr_range = np.arange(-10, 35, 0.5)
    overhead = 0.17   # pilots + control overhead fraction
    re_per_ms = n_rb * 84 * 2 * (1 - overhead)
    throughputs_mbps = []

    for snr in snr_range:
        mcs_idx  = select_mcs(snr)
        mod_ord, code_rate, _ = MCS_TABLE[mcs_idx]
        bits_per_re = mod_ord * code_rate
        tp = re_per_ms * bits_per_re * 1000 / 1e6  # Mbps
        throughputs_mbps.append(tp)

    return snr_range, np.array(throughputs_mbps)


# ===========================================================================
# 5. Peak throughput calculator
# ===========================================================================

def peak_throughput_lte(bandwidth_mhz: float = 20.0,
                         mimo_layers: int = 2,
                         modulation_order: int = 6,
                         code_rate: float = 0.93,
                         overhead_fraction: float = 0.17) -> dict:
    """
    Calculate LTE peak DL throughput.

    Parameters
    ----------
    bandwidth_mhz    : channel bandwidth
    mimo_layers      : number of spatial streams
    modulation_order : bits per symbol (2=QPSK, 4=16QAM, 6=64QAM)
    code_rate        : turbo code rate
    overhead_fraction: fraction of REs used for pilots/control
    """
    n_rb     = BANDWIDTH_CONFIG[bandwidth_mhz][0]
    re_per_subframe = n_rb * 12 * 14   # all REs in 1 ms
    usable_re       = re_per_subframe * (1 - overhead_fraction)
    bits_per_re     = modulation_order * code_rate
    tp_mbps         = usable_re * bits_per_re * mimo_layers * 1000 / 1e6

    return {
        "bandwidth_mhz":   bandwidth_mhz,
        "n_rb":            n_rb,
        "re_per_subframe": int(re_per_subframe),
        "usable_re":       int(usable_re),
        "bits_per_re":     round(bits_per_re, 2),
        "mimo_layers":     mimo_layers,
        "peak_mbps":       round(tp_mbps, 1),
    }


# ===========================================================================
# 6. End-to-end OFDMA transceiver with channel and pilot estimation
# ===========================================================================

def ofdma_transceiver(n_rb: int = 6, snr_db: float = 20.0):
    """
    Minimal end-to-end OFDMA link:
      Tx: QAM modulate -> IFFT -> add CP
      Channel: frequency-selective multipath + AWGN
      Rx: remove CP -> FFT -> pilot-based LS channel estimation -> equalise -> demodulate
    Returns BER and EVM.
    """
    n_sc     = n_rb * 12     # active subcarriers
    # pick next power-of-two FFT size that comfortably holds the active subcarriers
    n_fft    = max(128, int(2 ** np.ceil(np.log2(n_sc + 4))))
    cp_len   = max(9, n_fft // 14)  # rough scaling of CP length with FFT size

    # Pilot positions: every 6th subcarrier starting at index 0
    pilot_sc = np.arange(0, n_sc, 6)
    data_sc  = np.array([i for i in range(n_sc) if i not in pilot_sc])

    # QPSK constellation
    bits_per_sym = 2
    n_data_sc    = len(data_sc)
    bits         = rng.integers(0, 2, size=n_data_sc * bits_per_sym)
    symbols_idx  = bits[0::2] * 2 + bits[1::2]
    qpsk_map     = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    tx_data_sym  = qpsk_map[symbols_idx]

    # Known pilot symbols (BPSK)
    pilot_sym = np.ones(len(pilot_sc), dtype=complex)

    # Assemble frequency-domain OFDM symbol
    freq_tx = np.zeros(n_fft, dtype=complex)
    sc_offset = (n_fft - n_sc) // 2   # centre the active subcarriers
    for i, sc in enumerate(data_sc):
        freq_tx[sc_offset + sc] = tx_data_sym[i]
    for i, sc in enumerate(pilot_sc):
        freq_tx[sc_offset + sc] = pilot_sym[i]

    # IFFT -> time domain -> add CP
    time_tx  = np.fft.ifft(freq_tx) * n_fft
    time_cp  = np.concatenate([time_tx[-cp_len:], time_tx])

    # Multipath channel (3 taps)
    h_taps  = np.array([1.0, 0.5 * np.exp(1j * 0.8), 0.3 * np.exp(1j * 1.5)])
    rx_conv = np.convolve(time_cp, h_taps)[:len(time_cp)]

    # AWGN
    sig_power  = np.mean(np.abs(rx_conv) ** 2)
    noise_var  = sig_power / (10 ** (snr_db / 10))
    noise      = (rng.normal(0, np.sqrt(noise_var / 2), size=len(rx_conv)) +
                  1j * rng.normal(0, np.sqrt(noise_var / 2), size=len(rx_conv)))
    rx_noisy   = rx_conv + noise

    # Remove CP -> FFT
    rx_no_cp  = rx_noisy[cp_len:]
    freq_rx   = np.fft.fft(rx_no_cp[:n_fft]) / n_fft

    # LS channel estimation at pilot positions
    rx_pilots      = freq_rx[sc_offset + pilot_sc]
    H_pilot_est    = rx_pilots / pilot_sym   # H = Y / X at pilots

    # Interpolate channel estimate across all active subcarriers
    H_est_all = np.interp(
        np.arange(n_sc),
        pilot_sc,
        H_pilot_est.real
    ) + 1j * np.interp(
        np.arange(n_sc),
        pilot_sc,
        H_pilot_est.imag
    )

    # Extract received data subcarriers and equalise (zero forcing)
    rx_data_sym = freq_rx[sc_offset + data_sc]
    H_data      = H_est_all[data_sc]
    eq_sym      = rx_data_sym / H_data

    # QPSK demodulation (nearest constellation point)
    def demod_qpsk(sym):
        dists = np.abs(sym[:, None] - qpsk_map[None, :]) ** 2
        return np.argmin(dists, axis=1)

    rx_idx  = demod_qpsk(eq_sym)
    rx_bits = np.stack([rx_idx // 2, rx_idx % 2], axis=1).flatten()

    ber     = np.mean(rx_bits != bits)

    # EVM (error vector magnitude)
    tx_sym_norm = tx_data_sym / np.max(np.abs(qpsk_map))
    eq_sym_norm = eq_sym / np.max(np.abs(eq_sym)) if np.max(np.abs(eq_sym)) > 0 else eq_sym
    evm_rms = np.sqrt(np.mean(np.abs(eq_sym - tx_data_sym) ** 2) / np.mean(np.abs(tx_data_sym) ** 2))

    return ber, evm_rms * 100, eq_sym, tx_data_sym


# ===========================================================================
# Plotting
# ===========================================================================

def plot_all():
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("4G LTE Air Interface", fontsize=15, fontweight="bold", y=0.99)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # ------------------------------------------------------------------
    # Panel 1: LTE resource grid (10 MHz, 1 subframe)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    grid_obj = LTEResourceGrid(bandwidth_mhz=10.0)

    # Assign some RBs to users for illustration
    user_colors = ["#4e9af1", "#f17c4e", "#4ef1a0", "#f1e24e", "#c44ef1"]
    user_rb_map = {0: range(0, 12), 1: range(12, 25),
                   2: range(25, 35), 3: range(35, 45), 4: range(45, 50)}
    for u, rbs in user_rb_map.items():
        for rb in rbs:
            grid_obj.allocate_rb_pair(rb, u)

    cmap_data = plt.get_cmap("Set2", 8)
    colors_grid = {
        0: "#cccccc",   # DATA (unallocated)
        1: "#ff6b6b",   # PILOT
        2: "#ffa500",   # CONTROL
    }
    for u in range(5):
        colors_grid[10 + u] = user_colors[u]

    img = np.zeros((grid_obj.n_sc, grid_obj.n_symbols, 3))
    c_lookup = {
        0: np.array([0.8, 0.8, 0.8]),
        1: np.array([1.0, 0.42, 0.42]),
        2: np.array([1.0, 0.65, 0.0]),
    }
    user_rgb = [
        np.array([0.31, 0.60, 0.95]),
        np.array([0.95, 0.49, 0.31]),
        np.array([0.31, 0.95, 0.63]),
        np.array([0.95, 0.89, 0.31]),
        np.array([0.77, 0.31, 0.95]),
    ]
    for sc in range(grid_obj.n_sc):
        for sym in range(grid_obj.n_symbols):
            val = grid_obj.grid[sc, sym]
            if val < 10:
                img[sc, sym] = c_lookup.get(val, np.array([0.5, 0.5, 0.5]))
            else:
                img[sc, sym] = user_rgb[val - 10]

    ax1.imshow(img, aspect="auto", origin="lower",
               extent=[0, grid_obj.n_symbols, 0, grid_obj.n_sc])
    ax1.set_xlabel("OFDM symbol index (14 symbols = 1 ms subframe)")
    ax1.set_ylabel("Subcarrier index (50 RBs × 12 = 600 sc)")
    ax1.set_title(f"LTE resource grid: 10 MHz, 1 subframe (1 ms)\n"
                  f"Overhead: {grid_obj.overhead_fraction()*100:.1f}%  "
                  f"Data REs: {grid_obj.count_data_res()}", fontsize=10)
    ax1.axvline(x=7, color="white", linewidth=1.0, linestyle="--", alpha=0.6)
    ax1.text(3.2, grid_obj.n_sc + 5, "Slot 0", ha="center", fontsize=8)
    ax1.text(10.2, grid_obj.n_sc + 5, "Slot 1", ha="center", fontsize=8)
    legend_patches = [
        mpatches.Patch(color="#ff6b6b", label="CRS pilot"),
        mpatches.Patch(color="#ffa500", label="PDCCH control"),
    ] + [mpatches.Patch(color=user_colors[u], label=f"User {u}") for u in range(5)]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=3)

    # ------------------------------------------------------------------
    # Panel 2: OFDMA scheduler — RB allocation map
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    n_rb_sched = 25
    rb_assign, bits_alloc, snr_map, bpr = simulate_ofdma_scheduler(
        n_users=5, n_rb=n_rb_sched)

    cmap = plt.get_cmap("Set2", 5)
    for rb in range(n_rb_sched):
        u = rb_assign[rb]
        ax2.barh(rb, 1, left=0, color=cmap(u), edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(0, n_rb_sched, 5))
    ax2.set_xlabel("Assigned (user colour)")
    ax2.set_ylabel("RB index")
    ax2.set_title("Proportional-fair scheduler\n(5 MHz, 5 users, 1 subframe)", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    for u in range(5):
        ax2.barh(0, 0, color=cmap(u), label=f"User {u}: {bits_alloc[u]/1e3:.1f} kb")
    ax2.legend(fontsize=7, loc="lower right")

    # ------------------------------------------------------------------
    # Panel 3: PAPR CCDF — OFDMA vs SC-FDMA
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    papr_o, papr_s = scfdma_vs_ofdma_papr(n_subcarriers=12, n_symbols=2000)

    def ccdf(data, bins=200):
        x   = np.linspace(data.min(), data.max(), bins)
        cdf = np.array([np.mean(data <= xi) for xi in x])
        return x, 1 - cdf

    x_o, c_o = ccdf(papr_o)
    x_s, c_s = ccdf(papr_s)
    ax3.semilogy(x_o, np.clip(c_o, 1e-3, 1), "tomato",  linewidth=2, label="OFDMA (downlink)")
    ax3.semilogy(x_s, np.clip(c_s, 1e-3, 1), "steelblue", linewidth=2, label="SC-FDMA (uplink)")
    ax3.set_xlabel("PAPR (dB)")
    ax3.set_ylabel("Prob(PAPR > x)")
    ax3.set_title("PAPR CCDF: OFDMA vs SC-FDMA\n(12 subcarriers, QPSK)", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which="both")
    ax3.set_ylim([1e-3, 1])

    # ------------------------------------------------------------------
    # Panel 4: Throughput vs SNR (link adaptation)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    snr_ax, tp_ax = throughput_vs_snr(n_rb=50)

    ax4.plot(snr_ax, tp_ax, "steelblue", linewidth=2.5)
    ax4.fill_between(snr_ax, tp_ax, alpha=0.15, color="steelblue")

    # Shade modulation regions
    for label, snr_lo, snr_hi, c in [
        ("QPSK",   -10, 7,  "#afd8f8"),
        ("16QAM",   7, 18,  "#a8e6cf"),
        ("64QAM",  18, 35,  "#f8c8a0"),
    ]:
        ax4.axvspan(snr_lo, snr_hi, alpha=0.12, color=c, label=label)
        ax4.text((snr_lo + snr_hi) / 2, 2, label, ha="center", fontsize=8, color="gray")

    ax4.set_xlabel("SNR (dB)")
    ax4.set_ylabel("Throughput (Mbps)")
    ax4.set_title("LTE DL throughput vs SNR\n(10 MHz, 1 antenna, link adaptation)", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-10, 35])

    # ------------------------------------------------------------------
    # Panel 5: Peak throughput table across bandwidths
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    bws     = [1.4, 3, 5, 10, 15, 20]
    rows    = []
    for bw in bws:
        r1 = peak_throughput_lte(bw, mimo_layers=1, modulation_order=6, code_rate=0.93)
        r2 = peak_throughput_lte(bw, mimo_layers=2, modulation_order=6, code_rate=0.93)
        rows.append([f"{bw} MHz", str(r1["n_rb"]),
                     f"{r1['peak_mbps']} Mbps",
                     f"{r2['peak_mbps']} Mbps"])

    table = ax5.table(
        cellText=rows,
        colLabels=["BW", "RBs", "1x1 SISO", "2x2 MIMO"],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax5.set_title("Peak DL throughput\n(64-QAM, r=15/16)", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 6: Constellation after channel + equalisation
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 0])
    ber, evm, eq_sym, tx_sym = ofdma_transceiver(n_rb=6, snr_db=25.0)
    ax6.scatter(eq_sym.real, eq_sym.imag, s=8, alpha=0.5, color="steelblue", label="Rx (equalised)")
    ideal = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    ax6.scatter(ideal.real, ideal.imag, s=80, marker="*", color="tomato", zorder=5, label="Ideal QPSK")
    ax6.set_title(f"QPSK constellation after multipath + equalisation\nBER={ber:.4f}  EVM={evm:.1f}%  SNR=25 dB", fontsize=9)
    ax6.set_xlabel("In-phase")
    ax6.set_ylabel("Quadrature")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([-2, 2])
    ax6.set_ylim([-2, 2])

    # ------------------------------------------------------------------
    # Panel 7: BER vs SNR for the OFDMA transceiver
    # ------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 1])
    snr_vals = np.arange(0, 35, 3)
    bers     = []
    evms     = []
    for s in snr_vals:
        b, e, _, _ = ofdma_transceiver(n_rb=12, snr_db=s)
        bers.append(max(b, 1e-5))
        evms.append(e)
    ax7.semilogy(snr_vals, bers, "steelblue", marker="o", linewidth=2, label="BER (QPSK, pilot LS eq)")
    ax7.set_xlabel("SNR (dB)")
    ax7.set_ylabel("BER")
    ax7.set_title("BER vs SNR: OFDMA transceiver\n(multipath channel, LS channel estimation)", fontsize=10)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------
    # Panel 8: LTE frame hierarchy diagram (text-based in plot)
    # ------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")

    hierarchy = [
        ("Radio frame",    "10 ms",   0.95, "#4e9af1"),
        ("Subframe (TTI)", "1 ms",    0.80, "#f17c4e"),
        ("Slot",           "0.5 ms",  0.65, "#4ef1a0"),
        ("OFDM symbol",    "71.4 μs", 0.50, "#f1e24e"),
        ("Resource Block", "180 kHz × 0.5 ms", 0.35, "#c44ef1"),
        ("Resource Elem.", "15 kHz × 71.4 μs", 0.20, "#4ef1f1"),
    ]
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    for name, duration, y, color in hierarchy:
        width = 0.85
        rect  = mpatches.FancyBboxPatch((0.05, y - 0.06), width, 0.11,
                                        boxstyle="round,pad=0.01",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=1.5)
        ax8.add_patch(rect)
        ax8.text(0.08, y, name, fontsize=9, va="center", fontweight="bold")
        ax8.text(0.90, y, duration, fontsize=8, va="center", ha="right", color="#444")

    ax8.set_title("LTE time-frequency hierarchy", fontsize=10)

    plt.savefig("phase2_topic3_lte_air_interface.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to phase2_topic3_lte_air_interface.png")


# ===========================================================================
# Console summary
# ===========================================================================

def print_summary():
    print("=" * 62)
    print("  4G LTE Air Interface: Key Results")
    print("=" * 62)

    print("\n1. Resource grid overhead (10 MHz, 1 antenna port):")
    g = LTEResourceGrid(bandwidth_mhz=10.0)
    print(f"   Total REs per subframe : {g.n_sc * g.n_symbols}")
    print(f"   Data REs               : {g.count_data_res()}")
    print(f"   Overhead fraction      : {g.overhead_fraction()*100:.1f}%")

    print("\n2. OFDMA proportional-fair scheduling (5 MHz, 5 users):")
    rb_assign, bits_alloc, _, _ = simulate_ofdma_scheduler(n_users=5, n_rb=25)
    for u in range(5):
        rbs = np.sum(rb_assign == u)
        print(f"   User {u}: {rbs:2d} RBs  {bits_alloc[u]/1e3:.2f} kb/ms")

    print("\n3. PAPR comparison (12 subcarriers, QPSK, 2000 symbols):")
    papr_o, papr_s = scfdma_vs_ofdma_papr(n_subcarriers=12, n_symbols=2000)
    print(f"   OFDMA  mean PAPR : {np.mean(papr_o):.2f} dB")
    print(f"   SC-FDMA mean PAPR: {np.mean(papr_s):.2f} dB")
    print(f"   Reduction        : {np.mean(papr_o) - np.mean(papr_s):.2f} dB")

    print("\n4. Peak DL throughput (64-QAM, r=15/16):")
    for bw in [5, 10, 20]:
        for layers in [1, 2]:
            r = peak_throughput_lte(bw, mimo_layers=layers)
            print(f"   {bw:4.1f} MHz, {layers}x{layers} MIMO: {r['peak_mbps']:6.1f} Mbps")

    print("\n5. MCS selection at key SNR points:")
    for snr in [-5, 0, 5, 10, 15, 20, 25, 28]:
        idx = select_mcs(snr)
        mod, cr, label = MCS_TABLE[idx]
        print(f"   SNR={snr:4d} dB -> {label:20s}  ({mod} bits/sym, rate={cr})")

    print("\n6. End-to-end OFDMA link (multipath + LS equalisation):")
    for snr in [5, 15, 25]:
        ber, evm, _, _ = ofdma_transceiver(snr_db=snr)
        print(f"   SNR={snr} dB: BER={ber:.4f}  EVM={evm:.1f}%")


if __name__ == "__main__":
    print_summary()
    plot_all()
