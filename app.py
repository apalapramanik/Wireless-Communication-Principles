"""
Wireless Communication Principles — Interactive Dashboard
Run:  streamlit run app.py
"""

import math
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from scipy.special import erfc as _erfc

# Allow importing from topic folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '02_Modulation_Techniques'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_Signal_Fundamentals'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '03_DSP'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '06_3G_CDMA_WCDMA'))

st.set_page_config(
    page_title="Wireless Comms Explorer",
    page_icon="📡",
    layout="wide",
)

# ── Sidebar navigation ────────────────────────────────────────────────
PAGES = [
    "🏠 Overview",
    "📡 FFT Explorer",
    "⚡ Aliasing Demo",
    "🗺️ Constellation Viewer",
    "📉 BER Curves",
    "📶 Path Loss & Link Budget",
    "🔀 OFDM Explorer",
    "🛰️ Mobile Network Architecture",
    "📻 CDMA / WCDMA",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("numpy · matplotlib · scipy — built from scratch")


# ═══════════════════════════════════════════════════════════════════════
# Helpers (self-contained so each page works independently)
# ═══════════════════════════════════════════════════════════════════════

def gray_code(n_bits):
    return [i ^ (i >> 1) for i in range(2 ** n_bits)]

def qam_constellation(M):
    K = int(np.sqrt(M))
    bits_per_axis = int(np.log2(K))
    levels = np.arange(-(K - 1), K, 2, dtype=float)
    gray   = gray_code(bits_per_axis)
    symbols, labels = [], []
    for q_idx in gray[::-1]:
        for i_idx in gray:
            symbols.append(complex(levels[i_idx], levels[q_idx]))
            labels.append(
                format(q_idx, f'0{bits_per_axis}b') +
                format(i_idx, f'0{bits_per_axis}b')
            )
    symbols = np.array(symbols)
    symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2))
    return symbols, labels

def awgn(symbols, snr_db, rng):
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(0.5 / snr_lin)
    noise = noise_std * (rng.standard_normal(len(symbols)) +
                         1j * rng.standard_normal(len(symbols)))
    return symbols + noise

def simulate_ber(M, snr_db, n_symbols=30_000):
    bps = int(np.log2(M))
    constellation, labels = qam_constellation(M)
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, n_symbols * bps)
    bit_matrix = bits.reshape(n_symbols, bps)
    powers  = 2 ** np.arange(bps - 1, -1, -1)
    tx_idx  = (bit_matrix * powers).sum(axis=1).astype(int)
    tx_syms = constellation[tx_idx]
    rx_syms = awgn(tx_syms, snr_db, rng)
    distances = np.abs(rx_syms[:, None] - constellation[None, :])
    rx_idx  = np.argmin(distances, axis=1)
    errors  = sum(
        sum(a != b for a, b in zip(labels[t], labels[r]))
        for t, r in zip(tx_idx, rx_idx)
    )
    return errors / (n_symbols * bps)

def q_func(x):
    return np.array([0.5 * math.erfc(v / math.sqrt(2)) for v in np.atleast_1d(x)])

def ber_theory(M, snr_db_arr):
    snr_lin = 10 ** (np.asarray(snr_db_arr) / 10)
    k       = np.log2(M)
    eb_n0   = snr_lin / k
    arg     = np.sqrt(3 * k * eb_n0 / (M - 1))
    return (4 / k) * (1 - 1 / np.sqrt(M)) * q_func(arg)

def fspl_db(d, f):
    return 20 * np.log10(d) + 20 * np.log10(f) - 147.55

def log_distance_pl(d, d0, pl_d0, n):
    return pl_d0 + 10 * n * np.log10(d / d0)

def firwin_lpf(n_taps, cutoff_hz, fs):
    fc = cutoff_hz / fs
    M  = n_taps - 1
    n  = np.arange(n_taps)
    h  = 2 * fc * np.sinc(2 * fc * (n - M / 2))
    h *= np.hamming(n_taps)
    h /= h.sum()
    return h


# ═══════════════════════════════════════════════════════════════════════
# OFDM Helpers
# ═══════════════════════════════════════════════════════════════════════

def _ofdm_qam_const(M):
    m = int(np.sqrt(M))
    levels = np.arange(-(m - 1), m, 2, dtype=float)
    I, Q = np.meshgrid(levels, levels)
    pts = (I + 1j * Q).flatten()
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts

def _ofdm_qam_mod(bits, M):
    const = _ofdm_qam_const(M)
    bps   = int(np.log2(M))
    n_sym = len(bits) // bps
    words = bits[:n_sym * bps].reshape(-1, bps)
    idx   = np.array([int("".join(map(str, row)), 2) for row in words])
    return const[idx]

def _ofdm_qam_demod(rx_syms, M):
    const = _ofdm_qam_const(M)
    bps   = int(np.log2(M))
    idx   = np.argmin(np.abs(rx_syms[:, None] - const[None, :]), axis=1)
    bits  = []
    for i in idx:
        bits.extend([int(b) for b in format(i, f"0{bps}b")])
    return np.array(bits)

def _ofdm_mod(syms, N_fft, N_cp, N_active):
    freq = np.zeros(N_fft, dtype=complex)
    half = N_active // 2
    freq[1 : half + 1]         = syms[:half]
    freq[N_fft - half : N_fft] = syms[half:]
    td = np.fft.ifft(freq)
    return np.concatenate([td[-N_cp:], td])

def _ofdm_demod(rx, N_fft, N_cp, N_active):
    fd   = np.fft.fft(rx[N_cp : N_cp + N_fft])
    half = N_active // 2
    return np.concatenate([fd[1 : half + 1], fd[N_fft - half : N_fft]]), fd

def _ofdm_multipath(tx, delays, gains):
    max_d = max(delays)
    out   = np.zeros(len(tx) + max_d, dtype=complex)
    for d, g in zip(delays, gains):
        out[d : d + len(tx)] += g * tx
    return out[: len(tx)]

def _ofdm_awgn(signal, snr_db, rng):
    pwr = np.mean(np.abs(signal) ** 2)
    std = np.sqrt(pwr / (2 * 10 ** (snr_db / 10)))
    return signal + std * (rng.standard_normal(signal.shape) +
                           1j * rng.standard_normal(signal.shape))

def _ofdm_h_active(delays, gains, N_fft, N_active):
    h    = np.zeros(N_fft, dtype=complex)
    for d, g in zip(delays, gains):
        h[d] = g
    H    = np.fft.fft(h, N_fft)
    half = N_active // 2
    return np.concatenate([H[1 : half + 1], H[N_fft - half : N_fft]])

@st.cache_data
def _ofdm_run_link(n_sym, M, N_fft, N_cp, N_active, snr_db,
                   delays_t, gains_re_t, gains_im_t, use_eq, seed):
    delays = list(delays_t)
    gains  = [r + 1j * i for r, i in zip(gains_re_t, gains_im_t)]
    rng    = np.random.default_rng(seed)
    bps    = int(np.log2(M))
    H_act  = _ofdm_h_active(delays, gains, N_fft, N_active)
    tx_bits = rng.integers(0, 2, n_sym * N_active * bps)
    all_rx, rx_bits = [], []
    for i in range(n_sym):
        b        = tx_bits[i * N_active * bps : (i + 1) * N_active * bps]
        s        = _ofdm_qam_mod(b, M)
        td       = _ofdm_mod(s, N_fft, N_cp, N_active)
        td       = _ofdm_multipath(td, delays, gains)
        td       = _ofdm_awgn(td, snr_db, rng)
        rs, _    = _ofdm_demod(td, N_fft, N_cp, N_active)
        if use_eq:
            rs = rs / H_act
        all_rx.append(rs)
        rx_bits.extend(_ofdm_qam_demod(rs, M))
    rx_syms = np.concatenate(all_rx)
    rx_arr  = np.array(rx_bits)
    ber     = np.sum(tx_bits != rx_arr[: len(tx_bits)]) / len(tx_bits)
    return float(ber), rx_syms.real.tolist(), rx_syms.imag.tolist()


# ═══════════════════════════════════════════════════════════════════════
# Page: Overview
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Wireless Communication Principles")
    st.markdown(
        "An interactive companion to the repo — explore every core concept with live controls."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Topics covered", "8")
    col2.metric("Python files", "20+")
    col3.metric("External deps", "numpy · matplotlib · scipy")

    st.markdown("---")
    st.markdown("""
| Page | What you can explore |
|------|----------------------|
| 📡 **FFT Explorer** | Build any composite signal, see its spectrum live |
| ⚡ **Aliasing Demo** | Slide the sampling rate below Nyquist and watch aliasing happen |
| 🗺️ **Constellation Viewer** | Pick modulation order, add noise, see symbols scatter |
| 📉 **BER Curves** | Theory vs simulation — how SNR determines error rate |
| 📶 **Path Loss & Link Budget** | Tune distance, frequency, antenna gains — PASS or FAIL |
| 🔀 **OFDM Explorer** | Build a multipath channel, equalize it, measure BER vs SNR |
| 🛰️ **Mobile Network Architecture** | Walk through the 4G LTE attach procedure, message by message |
| 📻 **CDMA / WCDMA** | Spread codes, near-far, power control loop, RAKE multipath combining |
    """)

    st.info("Use the sidebar to navigate between pages.")


# ═══════════════════════════════════════════════════════════════════════
# Page: FFT Explorer
# ═══════════════════════════════════════════════════════════════════════
elif page == "📡 FFT Explorer":
    st.title("📡 FFT Explorer")
    st.markdown(
        "Build a signal from up to 3 tones. The FFT decomposes it back into its exact components."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Signal components")
        f1 = st.slider("Tone 1 frequency (Hz)", 10, 450, 50)
        a1 = st.slider("Tone 1 amplitude",      0.1, 2.0, 1.0, step=0.1)
        f2 = st.slider("Tone 2 frequency (Hz)", 10, 450, 120)
        a2 = st.slider("Tone 2 amplitude",      0.0, 2.0, 0.5, step=0.1)
        f3 = st.slider("Tone 3 frequency (Hz)", 10, 450, 300)
        a3 = st.slider("Tone 3 amplitude",      0.0, 2.0, 0.3, step=0.1)
        add_noise = st.checkbox("Add noise", value=False)
        noise_std = st.slider("Noise std dev", 0.0, 1.0, 0.2, step=0.05,
                              disabled=not add_noise)

    f_s = 1000
    N   = 1000
    t   = np.linspace(0, 1.0, N, endpoint=False)
    x   = a1*np.cos(2*np.pi*f1*t) + a2*np.cos(2*np.pi*f2*t) + a3*np.cos(2*np.pi*f3*t)
    if add_noise:
        x += np.random.default_rng(0).normal(0, noise_std, N)

    X     = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1/f_s)
    mag   = np.abs(X) * 2 / N

    with col_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(t[:150], x[:150], color='steelblue', lw=1)
        ax1.set_title("Time Domain (first 150 ms)", fontsize=11)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2.stem(freqs, mag, markerfmt='C1o', linefmt='C1-', basefmt='k-')
        for f, a, c in [(f1,a1,'C0'), (f2,a2,'C2'), (f3,a3,'C3')]:
            if a > 0:
                ax2.axvline(f, color=c, ls='--', alpha=0.6, label=f'{f} Hz')
        ax2.set_title("Frequency Domain (FFT)", fontsize=11)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim([0, 500])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    top3 = freqs[np.argsort(mag)[-3:][::-1]].astype(int)
    st.success(f"FFT detected top peaks at: **{top3[0]} Hz, {top3[1]} Hz, {top3[2]} Hz**")


# ═══════════════════════════════════════════════════════════════════════
# Page: Aliasing Demo
# ═══════════════════════════════════════════════════════════════════════
elif page == "⚡ Aliasing Demo":
    st.title("⚡ Aliasing Demo")
    st.markdown(
        "Slide the sampling rate below the **Nyquist limit** (2 × signal frequency) "
        "and watch the sampler misidentify the signal."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        f_signal = st.slider("Signal frequency (Hz)", 10, 200, 100)
        f_s      = st.slider("Sampling rate (Hz)", 20, 1000, 300)
        nyquist  = f_s / 2
        ok       = f_s >= 2 * f_signal

        st.metric("Nyquist limit", f"{nyquist:.0f} Hz")
        st.metric("Signal freq",   f"{f_signal} Hz")
        if ok:
            st.success(f"✅ f_s = {f_s} Hz ≥ 2×{f_signal} — no aliasing")
        else:
            alias = abs(f_signal - round(f_signal / f_s) * f_s)
            st.error(f"❌ ALIASED — perceived as **{alias:.0f} Hz**")
            st.caption(f"Formula: |{f_signal} − {round(f_signal/f_s)}×{f_s}| = {alias:.0f}")

    t_fine = np.linspace(0, 0.05, 5000)
    t_s    = np.arange(0, 0.05, 1 / f_s)
    x_s    = np.cos(2 * np.pi * f_signal * t_s)

    X    = np.fft.rfft(x_s, n=len(t_s))
    freq = np.fft.rfftfreq(len(t_s), 1 / f_s)
    peak = freq[np.argmax(np.abs(X))]

    with col_plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t_fine, np.cos(2*np.pi*f_signal*t_fine),
                color='steelblue', alpha=0.3, lw=1.5, label=f'True {f_signal} Hz signal')
        ax.plot(t_s, x_s, 'ro-', markersize=5, lw=1.2, label=f'Samples @ {f_s} Hz')

        if not ok:
            alias = abs(f_signal - round(f_signal / f_s) * f_s)
            ax.plot(t_fine, np.cos(2*np.pi*alias*t_fine),
                    color='orange', lw=1.5, alpha=0.8,
                    label=f'Perceived alias: {alias:.0f} Hz')

        ax.set_title(
            f'f_signal={f_signal} Hz  |  f_s={f_s} Hz  |  '
            f'Perceives {peak:.0f} Hz  {"✅" if ok else "❌ ALIASED"}',
            fontsize=11
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.5, 1.8])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Page: Constellation Viewer
# ═══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Constellation Viewer":
    st.title("🗺️ Constellation Viewer")
    st.markdown(
        "Each dot is a symbol — a unique (I, Q) point the transmitter can send. "
        "Add noise to see the cloud of received points and where errors happen."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        M       = st.selectbox("Modulation order M", [4, 16, 64, 256], index=1)
        snr_db  = st.slider("SNR (dB)", 0, 35, 20)
        n_syms  = st.slider("Symbols to plot", 500, 5000, 2000, step=500)
        show_tx = st.checkbox("Show TX points", value=True)
        show_rx = st.checkbox("Show RX (noisy) points", value=True)
        show_labels = st.checkbox("Show bit labels", value=M <= 16)

        bps = int(np.log2(M))
        st.metric("Bits per symbol", bps)
        st.metric("Constellation points", M)

    constellation, labels = qam_constellation(M)
    rng = np.random.default_rng(0)
    bits    = rng.integers(0, 2, n_syms * bps)
    bm      = bits.reshape(n_syms, bps)
    powers  = 2 ** np.arange(bps - 1, -1, -1)
    tx_idx  = (bm * powers).sum(axis=1).astype(int)
    tx_syms = constellation[tx_idx]
    rx_syms = awgn(tx_syms, snr_db, rng)

    distances = np.abs(rx_syms[:, None] - constellation[None, :])
    rx_idx    = np.argmin(distances, axis=1)
    errors    = np.sum(tx_idx != rx_idx)
    ber_sim   = sum(
        sum(a != b for a, b in zip(labels[t], labels[r]))
        for t, r in zip(tx_idx, rx_idx)
    ) / (n_syms * bps)

    with col_ctrl:
        st.metric("Symbol errors", f"{errors} / {n_syms}")
        st.metric("Simulated BER", f"{ber_sim:.3e}")

    with col_plot:
        fig, ax = plt.subplots(figsize=(7, 7))

        if show_rx:
            ax.scatter(rx_syms.real, rx_syms.imag,
                       s=4, alpha=0.3, color='steelblue', label='RX (noisy)')
        if show_tx:
            ax.scatter(constellation.real, constellation.imag,
                       s=60, color='red', zorder=5, label='TX (ideal)')
        if show_labels and M <= 64:
            for sym, lbl in zip(constellation, labels):
                ax.annotate(lbl, (sym.real, sym.imag),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6 if M <= 16 else 5)

        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_title(f'{M}-{"QPSK" if M==4 else "QAM"}  SNR={snr_db} dB  '
                     f'({bps} bits/symbol)', fontsize=12)
        ax.set_xlabel('In-phase  I')
        ax.set_ylabel('Quadrature  Q')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Page: BER Curves
# ═══════════════════════════════════════════════════════════════════════
elif page == "📉 BER Curves":
    st.title("📉 BER vs SNR Curves")
    st.markdown(
        "Theory lines (exact formula) vs simulated dots. "
        "The dashed lines mark 5G NR target BER levels."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        schemes  = st.multiselect(
            "Modulation schemes",
            [4, 16, 64, 256],
            default=[4, 16, 64],
            format_func=lambda m: f"{m}-{'QPSK' if m==4 else 'QAM'}"
        )
        snr_max  = st.slider("Max SNR (dB)", 20, 40, 35)
        run_sim  = st.checkbox("Show simulation dots", value=True)
        n_syms   = st.select_slider(
            "Simulation symbols",
            options=[10_000, 30_000, 100_000],
            value=30_000,
            disabled=not run_sim
        )

    snr_range = np.arange(0, snr_max + 1, 0.5)
    colors    = {4: 'tab:blue', 16: 'tab:orange', 64: 'tab:green', 256: 'tab:red'}
    names     = {4: 'QPSK', 16: '16-QAM', 64: '64-QAM', 256: '256-QAM'}

    fig, ax = plt.subplots(figsize=(9, 6))

    if schemes:
        for M in schemes:
            th = ber_theory(M, snr_range)
            ax.semilogy(snr_range, th, '-', color=colors[M],
                        lw=2, label=f'{names[M]} — theory')

            if run_sim:
                sim_snrs = range(0, snr_max + 1, 3)
                with st.spinner(f"Simulating {names[M]}…"):
                    sim_bers = [max(simulate_ber(M, s, n_syms), 1e-7) for s in sim_snrs]
                ax.semilogy(list(sim_snrs), sim_bers, 'o',
                            color=colors[M], ms=5, label=f'{names[M]} — sim')

    ax.axhline(1e-3, color='gray', ls='--', alpha=0.6, label='10⁻³ (pre-coding target)')
    ax.axhline(1e-5, color='gray', ls=':',  alpha=0.6, label='10⁻⁵ (after coding)')
    ax.set_xlabel('SNR per symbol (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs SNR — Square QAM, AWGN, Gray Coded', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-7, 1])
    ax.set_xlim([0, snr_max])

    with col_plot:
        st.pyplot(fig)
        plt.close(fig)

    if schemes:
        with col_ctrl:
            st.markdown("**Required SNR @ BER = 10⁻³**")
            for M in schemes:
                lo, hi = 0.0, 50.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if ber_theory(M, np.array([mid]))[0] > 1e-3:
                        lo = mid
                    else:
                        hi = mid
                st.metric(names[M], f"{(lo+hi)/2:.1f} dB")


# ═══════════════════════════════════════════════════════════════════════
# Page: Path Loss & Link Budget
# ═══════════════════════════════════════════════════════════════════════
elif page == "📶 Path Loss & Link Budget":
    st.title("📶 Path Loss & Link Budget")

    tab1, tab2 = st.tabs(["Path Loss Curves", "Link Budget Calculator"])

    # ── Tab 1: Path Loss ─────────────────────────────────────────────
    with tab1:
        st.markdown("Compare FSPL across frequencies and add log-normal shadowing.")
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            freq_choice = st.multiselect(
                "Frequencies",
                ["700 MHz (4G)", "3.5 GHz (5G FR1)", "28 GHz (mmWave)"],
                default=["700 MHz (4G)", "3.5 GHz (5G FR1)", "28 GHz (mmWave)"]
            )
            d_max   = st.slider("Max distance (km)", 1, 20, 5)
            show_shadow = st.checkbox("Add shadowing (σ=8 dB)", value=False)
            n_val   = st.slider("Path loss exponent n", 2.0, 5.0, 3.5, step=0.1)

        freq_map = {
            "700 MHz (4G)":      (700e6,  "royalblue"),
            "3.5 GHz (5G FR1)":  (3.5e9,  "forestgreen"),
            "28 GHz (mmWave)":   (28e9,   "crimson"),
        }
        distances = np.logspace(1, np.log10(d_max * 1000), 400)

        with col_plot:
            fig, ax = plt.subplots(figsize=(9, 5))
            for label in freq_choice:
                f, color = freq_map[label]
                pl = fspl_db(distances, f)
                ax.plot(distances/1000, pl, color=color, lw=2, label=label)
                if show_shadow:
                    pl_d0  = fspl_db(100, f)
                    pl_ld  = log_distance_pl(distances, 100, pl_d0, n_val)
                    shadow = np.random.default_rng(7).normal(0, 8, len(distances))
                    ax.plot(distances/1000, pl_ld + shadow,
                            color=color, lw=0.8, alpha=0.4)

            ax.axhline(140, color='gray', ls='--', alpha=0.5,
                       label='Max link budget ~140 dB')
            ax.set_xscale('log')
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Path Loss (dB)')
            ax.set_title('Free Space Path Loss vs Distance')
            ax.legend(fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 2: Link Budget ───────────────────────────────────────────
    with tab2:
        st.markdown("Adjust all parameters — see PASS/FAIL and link margin live.")

        c1, c2 = st.columns(2)
        with c1:
            tx_power  = st.slider("TX Power (dBm)",        20, 50, 43)
            tx_gain   = st.slider("TX Antenna Gain (dBi)", 0,  30, 15)
            rx_gain   = st.slider("RX Antenna Gain (dBi)", 0,  20,  0)
            misc_loss = st.slider("Misc Losses (dB)",       0,  30,  3)
        with c2:
            freq_sel  = st.selectbox("Frequency", ["700 MHz", "3.5 GHz", "28 GHz"])
            distance  = st.slider("Distance (m)", 50, 5000, 500, step=50)
            bw_mhz    = st.slider("Bandwidth (MHz)", 5, 400, 100, step=5)
            nf        = st.slider("Noise Figure (dB)", 3, 15, 7)
            req_snr   = st.slider("Required SNR (dB)", 5, 25, 10)
            n_ple     = st.slider("Path Loss Exponent", 2.0, 5.0, 3.5, step=0.1)

        freq_hz_map = {"700 MHz": 700e6, "3.5 GHz": 3.5e9, "28 GHz": 28e9}
        freq_hz = freq_hz_map[freq_sel]

        eirp      = tx_power + tx_gain
        pl_d0     = fspl_db(100, freq_hz)
        pl_db     = log_distance_pl(distance, 100, pl_d0, n_ple)
        rx_power  = eirp - pl_db + rx_gain - misc_loss
        noise_floor = -174 + 10 * np.log10(bw_mhz * 1e6)
        rx_sens   = noise_floor + nf + req_snr
        margin    = rx_power - rx_sens

        st.markdown("---")
        cols = st.columns(4)
        cols[0].metric("EIRP",          f"{eirp} dBm")
        cols[1].metric("Path Loss",     f"{pl_db:.1f} dB")
        cols[2].metric("Rx Power",      f"{rx_power:.1f} dBm")
        cols[3].metric("Rx Sensitivity",f"{rx_sens:.1f} dBm")

        cols2 = st.columns(2)
        cols2[0].metric("Noise Floor",  f"{noise_floor:.1f} dBm")
        cols2[1].metric("Link Margin",  f"{margin:.1f} dB",
                        delta=f"{'PASS ✅' if margin > 0 else 'FAIL ❌'}")

        if margin > 0:
            st.success(f"✅ Link margin = **{margin:.1f} dB** — link is viable")
        else:
            st.error(f"❌ Link margin = **{margin:.1f} dB** — link fails  "
                     f"(need {-margin:.1f} dB more gain or {distance}m → "
                     f"{distance * 10**(margin / (10*n_ple)):.0f}m distance)")


# ═══════════════════════════════════════════════════════════════════════
# Page: OFDM Explorer
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔀 OFDM Explorer":
    st.title("🔀 OFDM Explorer")
    st.markdown(
        "A 5G NR-like OFDM link — 128-point FFT, 72 active subcarriers, 16-sample cyclic prefix. "
        "Explore how multipath distorts the constellation and how a zero-forcing equalizer undoes it."
    )

    _N_FFT, _N_CP, _N_ACTIVE = 128, 16, 72
    _DELAYS = [0, 4, 9]
    _GAINS_DEFAULT = [1.0, 0.6 * np.exp(1j * 0.8), 0.3 * np.exp(-1j * 1.2)]

    tab1, tab2, tab3 = st.tabs(["📶 Channel Model", "🗺️ Constellation", "📉 BER vs SNR"])

    # ── Tab 1: Channel Model ─────────────────────────────────────────
    with tab1:
        st.markdown(
            "Adjust the 3-tap channel gains and phases. "
            "The CP must be longer than the max tap delay or ISI leaks in."
        )
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            st.subheader("Tap controls")
            g1   = st.slider("Tap 1 (delay = 0) gain",  0.1, 2.0, 1.0, step=0.05)
            g2   = st.slider("Tap 2 (delay = 4) gain",  0.0, 1.5, 0.6, step=0.05)
            phi2 = st.slider("Tap 2 phase (rad)",        0.0, 6.28, 0.8, step=0.05)
            g3   = st.slider("Tap 3 (delay = 9) gain",  0.0, 1.0, 0.3, step=0.05)
            phi3 = st.slider("Tap 3 phase (rad)",        0.0, 6.28, 1.2, step=0.05)
            cp_disp = st.slider("CP length (display only)", 4, 32, _N_CP)

            gains_ch = [g1, g2 * np.exp(1j * phi2), g3 * np.exp(-1j * phi3)]
            max_d    = max(_DELAYS)
            if cp_disp >= max_d:
                st.success(f"✅ CP ({cp_disp}) ≥ max delay ({max_d}) — no ISI")
            else:
                st.error(f"❌ CP ({cp_disp}) < max delay ({max_d}) — ISI leaks in")
            st.markdown("---")
            st.markdown(f"**Subcarrier count:** {_N_ACTIVE} active / {_N_FFT} total")
            st.markdown(f"**Guard subcarriers:** {_N_FFT - _N_ACTIVE - 1} (edge roll-off)")

        h_vec = np.zeros(32, dtype=complex)
        for d, g in zip(_DELAYS, gains_ch):
            h_vec[d] = g
        H_full   = np.fft.fft(h_vec, _N_FFT)
        freqs_sh = np.fft.fftshift(np.fft.fftfreq(_N_FFT))
        H_sh     = np.fft.fftshift(H_full)

        with col_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

            ax1.stem(np.arange(len(h_vec)), np.abs(h_vec),
                     markerfmt="C0o", linefmt="C0-", basefmt="k-")
            ax1.axvline(cp_disp, color="red", ls="--", lw=1.5,
                        label=f"CP = {cp_disp} samples")
            ax1.set(title="Channel Impulse Response  |h[n]|",
                    xlabel="Delay (samples)", ylabel="|h|")
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2.plot(freqs_sh, 20 * np.log10(np.abs(H_sh) + 1e-10),
                     color="steelblue", lw=1.5)
            ax2.set(title="Channel Frequency Response  |H(f)|  dB",
                    xlabel="Normalized frequency (−0.5 … +0.5)", ylabel="dB")
            ax2.grid(True, alpha=0.3)

            plt.suptitle("3-Tap Multipath Channel", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "**Key insight:** each path adds a delayed copy of the signal. "
            "At some subcarrier frequencies the copies add constructively (peaks); "
            "at others destructively (notches). "
            "After the FFT, each subcarrier *k* sees a single scalar H_k — "
            "this is what the ZF equalizer divides out."
        )

    # ── Tab 2: Constellation ─────────────────────────────────────────
    with tab2:
        st.markdown(
            "Each OFDM symbol passes through the 3-tap channel. "
            "Without equalization every subcarrier lands at a different rotated position. "
            "The ZF equalizer divides by H_k and collapses the clouds back."
        )
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            M_c    = st.selectbox("Modulation order M", [4, 16, 64], index=1, key="oc_M")
            snr_c  = st.slider("SNR (dB)", 0, 35, 25, key="oc_snr")
            nsym_c = st.slider("OFDM symbols to simulate", 10, 100, 40, step=10, key="oc_nsym")

        gains_c = _GAINS_DEFAULT
        ber_eq,   re_eq,   im_eq   = _ofdm_run_link(
            nsym_c, M_c, _N_FFT, _N_CP, _N_ACTIVE, snr_c,
            tuple(_DELAYS),
            tuple(g.real for g in gains_c), tuple(g.imag for g in gains_c),
            True, 42)
        ber_noeq, re_noeq, im_noeq = _ofdm_run_link(
            nsym_c, M_c, _N_FFT, _N_CP, _N_ACTIVE, snr_c,
            tuple(_DELAYS),
            tuple(g.real for g in gains_c), tuple(g.imag for g in gains_c),
            False, 42)

        const_ref = _ofdm_qam_const(M_c)

        with col_ctrl:
            st.markdown("---")
            st.metric("BER — ZF equalizer",  f"{ber_eq:.4f}")
            st.metric("BER — no equalizer",  f"{ber_noeq:.3f}")
            if ber_eq > 0:
                st.metric("Improvement", f"{ber_noeq / ber_eq:.0f}×")

        with col_plot:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            label = "QPSK" if M_c == 4 else f"{M_c}-QAM"

            axes[0].scatter(const_ref.real, const_ref.imag, s=80, color="black", zorder=5)
            axes[0].set(title=f"{label} — Ideal", aspect="equal")
            axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(re_noeq, im_noeq, s=3, alpha=0.35, color="crimson")
            axes[1].scatter(const_ref.real, const_ref.imag,
                            s=50, color="black", zorder=5, marker="x")
            axes[1].set(title=f"No equalizer\nBER = {ber_noeq:.3f}", aspect="equal")
            axes[1].axhline(0, color="k", lw=0.5); axes[1].axvline(0, color="k", lw=0.5)
            axes[1].grid(True, alpha=0.3)

            axes[2].scatter(re_eq, im_eq, s=3, alpha=0.35, color="steelblue")
            axes[2].scatter(const_ref.real, const_ref.imag,
                            s=50, color="black", zorder=5, marker="x")
            axes[2].set(title=f"ZF equalizer\nBER = {ber_eq:.4f}", aspect="equal")
            axes[2].axhline(0, color="k", lw=0.5); axes[2].axvline(0, color="k", lw=0.5)
            axes[2].grid(True, alpha=0.3)

            plt.suptitle("OFDM Constellation — Multipath + ZF Equalization", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 3: BER vs SNR ────────────────────────────────────────────
    with tab3:
        st.markdown(
            "With ZF equalization the BER curve tracks theoretical AWGN — "
            "confirming the CP + FFT + one-tap division fully neutralizes multipath. "
            "Without it the BER floors: adding power stops helping."
        )
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            M_b       = st.selectbox("Modulation order M", [4, 16, 64], index=1, key="ob_M")
            snr_max_b = st.slider("Max SNR (dB)", 20, 35, 30, key="ob_snr_max")
            nsym_b    = st.select_slider(
                "OFDM symbols per SNR point",
                options=[10, 20, 30, 50], value=20, key="ob_nsym"
            )
            bps_b = int(np.log2(M_b))
            st.caption(f"~{nsym_b * _N_ACTIVE * bps_b:,} bits per point")

        gains_b  = _GAINS_DEFAULT
        snr_pts  = list(range(0, snr_max_b + 1, 3))
        bers_eq, bers_noeq = [], []

        with st.spinner("Simulating…"):
            for snr in snr_pts:
                b_eq,   _, _ = _ofdm_run_link(
                    nsym_b, M_b, _N_FFT, _N_CP, _N_ACTIVE, snr,
                    tuple(_DELAYS),
                    tuple(g.real for g in gains_b), tuple(g.imag for g in gains_b),
                    True, 7)
                b_noeq, _, _ = _ofdm_run_link(
                    nsym_b, M_b, _N_FFT, _N_CP, _N_ACTIVE, snr,
                    tuple(_DELAYS),
                    tuple(g.real for g in gains_b), tuple(g.imag for g in gains_b),
                    False, 7)
                bers_eq.append(max(b_eq, 1e-5))
                bers_noeq.append(max(b_noeq, 1e-5))

        snr_fine = np.linspace(0, snr_max_b, 200)
        ber_th   = ber_theory(M_b, snr_fine)
        label_b  = "QPSK" if M_b == 4 else f"{M_b}-QAM"

        with col_plot:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.semilogy(snr_pts, bers_noeq, "rs--", ms=6, lw=1.5,
                        label="Multipath, NO equalizer")
            ax.semilogy(snr_pts, bers_eq,   "bo-",  ms=6, lw=1.5,
                        label="Multipath + ZF equalizer")
            ax.semilogy(snr_fine, ber_th, "k--", alpha=0.5,
                        label=f"Theory ({label_b}, AWGN)")
            ax.axhline(1e-3, color="gray", ls=":", alpha=0.7,
                       label="BER = 10⁻³ target")
            ax.set(xlabel="SNR (dB)", ylabel="BER",
                   title=f"OFDM BER vs SNR — {label_b}, 3-tap Multipath Channel",
                   ylim=[1e-5, 1], xlim=[0, snr_max_b])
            ax.legend(fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_ctrl:
            st.markdown("---")
            cross = [snr_pts[i] for i, b in enumerate(bers_eq) if b < 1e-3]
            if cross:
                st.success(f"ZF equalizer hits BER < 10⁻³ at **{cross[0]} dB**")
            else:
                st.warning("Increase max SNR to reach BER < 10⁻³")


# ═══════════════════════════════════════════════════════════════════════
# Page: Mobile Network Architecture
# ═══════════════════════════════════════════════════════════════════════
elif page == "🛰️ Mobile Network Architecture":
    st.title("🛰️ Mobile Network Architecture")
    st.markdown(
        "Explore the **4G LTE attach procedure** — every signaling message exchanged "
        "when a phone joins the network, gets authenticated, and obtains an IP and a "
        "data tunnel to the internet."
    )

    tab1, tab2, tab3 = st.tabs(["🗺️ Network Topology", "📨 Attach Sequence", "📊 2G → 5G Evolution"])

    # ── Tab 1: Network Topology ─────────────────────────────────────
    with tab1:
        st.markdown(
            "The 4G **EPS** (Evolved Packet System) splits cleanly into the **RAN** "
            "(radio access) and the **EPC** (core), with separate control-plane and "
            "user-plane paths."
        )

        # node positions
        nodes = {
            "UE":  (0.05, 0.50),
            "eNB": (0.25, 0.50),
            "MME": (0.50, 0.78),
            "HSS": (0.78, 0.78),
            "SGW": (0.50, 0.22),
            "PGW": (0.78, 0.22),
            "Internet": (0.95, 0.22),
        }
        node_color = {
            "UE": "#4C72B0", "eNB": "#55A868",
            "MME": "#C44E52", "HSS": "#8172B3",
            "SGW": "#CCB974", "PGW": "#64B5CD",
            "Internet": "#888888",
        }
        # control-plane (red) and user-plane (blue) links
        cp_links = [
            ("UE", "eNB", "Uu (RRC/NAS)"),
            ("eNB", "MME", "S1-MME"),
            ("MME", "HSS", "S6a"),
            ("MME", "SGW", "S11"),
        ]
        up_links = [
            ("UE", "eNB", "Uu (DRB)"),
            ("eNB", "SGW", "S1-U (GTP)"),
            ("SGW", "PGW", "S5 (GTP)"),
            ("PGW", "Internet", "SGi"),
        ]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # plane bands
        ax.axhspan(0.62, 0.96, color="#fde7e7", alpha=0.6, zorder=0)
        ax.axhspan(0.04, 0.38, color="#e7f0fd", alpha=0.6, zorder=0)
        ax.text(0.99, 0.93, "Control plane", ha="right", fontsize=10,
                color="#a33", fontweight="bold")
        ax.text(0.99, 0.07, "User plane",    ha="right", fontsize=10,
                color="#338", fontweight="bold")

        for name, (x, y) in nodes.items():
            ax.add_patch(plt.Circle((x, y), 0.045,
                                    color=node_color[name], zorder=3))
            ax.text(x, y, name, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=4)

        def draw_link(a, b, label, color, offset=0.0):
            x1, y1 = nodes[a]
            x2, y2 = nodes[b]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=2, alpha=0.8), zorder=2)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + offset
            ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                    color=color, bbox=dict(facecolor="white",
                                           edgecolor="none", alpha=0.85))

        for a, b, lbl in cp_links:
            draw_link(a, b, lbl, "#c0392b", offset=0.025)
        for a, b, lbl in up_links:
            draw_link(a, b, lbl, "#2c5fa8", offset=-0.025)

        ax.set_title("4G LTE / EPS Reference Architecture", fontsize=12, pad=15)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
| Node | Plane | Role |
|------|-------|------|
| **UE** (User Equipment) | — | The phone — runs the air-interface stack and the SIM/USIM |
| **eNB** (evolved Node B) | RAN | LTE base station — RRC, scheduling, ciphering at AS layer |
| **MME** | Core CP | Mobility, authentication, attach orchestration — never touches user data |
| **HSS** | Core CP | Subscriber database — stores K, generates AKA vectors |
| **SGW** (Serving GW) | Core UP | Local user-plane anchor inside the operator network |
| **PGW** (PDN GW) | Core UP | Internet gateway — allocates UE IP, enforces QoS |
        """)

    # ── Tab 2: Attach Sequence ─────────────────────────────────────
    with tab2:
        st.markdown(
            "The full attach is **20+ messages across 6 phases**. Use the slider "
            "to step through the procedure and watch the message flow build up."
        )

        # (phase_idx, src, dst, label, plane)
        sequence = [
            (1, "UE",  "eNB", "RRC Connection Request",            "cp"),
            (1, "eNB", "UE",  "RRC Connection Setup",              "cp"),
            (1, "UE",  "eNB", "RRC Connection Setup Complete\n+ NAS Attach Request", "cp"),
            (2, "eNB", "MME", "Initial UE Message (S1-MME)",       "cp"),
            (3, "MME", "HSS", "Auth Info Request (S6a)",           "cp"),
            (3, "HSS", "MME", "Auth Info Answer (RAND, AUTN, XRES)", "cp"),
            (3, "MME", "UE",  "Authentication Request",            "cp"),
            (3, "UE",  "MME", "Authentication Response (RES)",     "cp"),
            (4, "MME", "UE",  "Security Mode Command",             "cp"),
            (4, "UE",  "MME", "Security Mode Complete",            "cp"),
            (5, "MME", "SGW", "Create Session Request (S11)",      "cp"),
            (5, "SGW", "PGW", "Create Session Request (S5)",       "cp"),
            (5, "PGW", "SGW", "Create Session Response\n(UE IP, PGW TEID)", "cp"),
            (5, "SGW", "MME", "Create Session Response (SGW TEID)", "cp"),
            (6, "MME", "eNB", "Initial Context Setup Request",     "cp"),
            (6, "eNB", "UE",  "RRC Reconfiguration\n(DRB setup, AS security)", "cp"),
            (6, "UE",  "eNB", "RRC Reconfiguration Complete",      "cp"),
            (6, "eNB", "MME", "Initial Context Setup Response\n(eNB TEID)", "cp"),
            (6, "UE",  "MME", "Attach Complete (NAS)",             "cp"),
            (7, "UE",  "Internet", "User-plane data via GTP tunnel", "up"),
        ]
        phase_names = {
            1: "RRC Connection Setup",
            2: "Initial UE Message",
            3: "EPS-AKA Authentication",
            4: "NAS Security Mode",
            5: "Default Bearer Setup",
            6: "Attach Accept + Radio Bearer",
            7: "User-Plane Data Flow",
        }

        col_ctrl, col_plot = st.columns([1, 3])
        with col_ctrl:
            n_steps = st.slider("Show messages 1 …", 1, len(sequence),
                                len(sequence), key="mna_n")
            cur_phase = sequence[n_steps - 1][0]
            st.metric("Current phase", f"{cur_phase}. {phase_names[cur_phase]}")
            st.metric("Messages shown", f"{n_steps} / {len(sequence)}")
            cp_count = sum(1 for s in sequence[:n_steps] if s[4] == "cp")
            up_count = n_steps - cp_count
            st.markdown(f"- **Control-plane:** {cp_count}")
            st.markdown(f"- **User-plane:**    {up_count}")

        # lifelines for each entity
        lanes = ["UE", "eNB", "MME", "HSS", "SGW", "PGW", "Internet"]
        lane_x = {n: i for i, n in enumerate(lanes)}

        with col_plot:
            fig, ax = plt.subplots(figsize=(11, max(6, n_steps * 0.45)))
            ax.set_xlim(-0.6, len(lanes) - 0.4)
            ax.set_ylim(-(n_steps + 1), 1)

            # lifelines
            for n, x in lane_x.items():
                ax.plot([x, x], [0, -(n_steps + 1)],
                        color="#aaa", lw=1, ls="--", zorder=1)
                ax.text(x, 0.4, n, ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

            for i, (ph, src, dst, label, plane) in enumerate(sequence[:n_steps]):
                y = -(i + 1)
                x1, x2 = lane_x[src], lane_x[dst]
                color = "#c0392b" if plane == "cp" else "#2c5fa8"
                ax.annotate("", xy=(x2, y), xytext=(x1, y),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            lw=1.6), zorder=3)
                mx = (x1 + x2) / 2
                ax.text(mx, y + 0.18, label, ha="center", va="bottom",
                        fontsize=8, color=color,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
                ax.text(-0.55, y, f"P{ph}", ha="right", va="center",
                        fontsize=8, color="#666")

            ax.axis("off")
            ax.set_title("LTE Attach — Message Sequence", fontsize=12, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "**Why the order matters:** authentication (Phase 3) must complete before "
            "any session is created. Security (Phase 4) must activate before the IMSI "
            "or session keys traverse the network unprotected. Only after the bearer "
            "exists end-to-end can user data flow."
        )

    # ── Tab 3: 2G → 5G Evolution ─────────────────────────────────
    with tab3:
        st.markdown(
            "Each generation rebuilt the architecture to push more intelligence to "
            "the edge and split control from user data more cleanly."
        )

        rows = [
            ("Air interface",  "TDMA/FDMA",  "WCDMA",     "OFDMA",       "OFDM/NR"),
            ("RAN node",       "BTS",        "Node B",    "eNB",         "gNB (RU+DU+CU)"),
            ("RAN controller", "BSC",        "RNC",       "None (flat)", "CU (logical)"),
            ("Core type",      "CS + PS",    "CS + PS",   "All-IP EPC",  "Cloud-native 5GC"),
            ("CP anchor",      "MSC/SGSN",   "SGSN",      "MME",         "AMF"),
            ("UP anchor",      "GGSN",       "GGSN",      "SGW/PGW",     "UPF"),
            ("Peak DL",        "~0.1 Mbps",  "~42 Mbps",  "~150 Mbps",   "~20 Gbps"),
            ("Latency (RTT)",  "~300 ms",    "~100 ms",   "~30 ms",      "~1 ms"),
            ("CP/UP split",    "No",         "No",        "Partial",     "Full (CUPS)"),
        ]
        table_md = "| Feature | 2G (GSM) | 3G (UMTS) | 4G (LTE) | 5G NR |\n"
        table_md += "|---|---|---|---|---|\n"
        for r in rows:
            table_md += f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"
        st.markdown(table_md)

        st.markdown("---")
        st.markdown("### Peak downlink throughput")
        gens   = ["2G (GSM)", "3G (UMTS)", "4G (LTE)", "5G NR"]
        speeds = [0.1, 42, 150, 20_000]   # Mbps
        gen_colors = ["#888", "#5a8db5", "#55a868", "#c44e52"]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(gens, speeds, color=gen_colors)
        ax.set_yscale("log")
        ax.set_ylabel("Peak DL throughput (Mbps, log scale)")
        ax.set_title("Generational leap — each step is roughly 100×")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        for bar, v in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v * 1.4,
                    f"{v:g} Mbps" if v < 1000 else f"{v/1000:g} Gbps",
                    ha="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.success(
            "**5G key shifts:** the gNB splits into Radio/Distributed/Centralized units "
            "(RU + DU + CU) for cloud RAN; the EPC becomes a service-based 5GC with the "
            "AMF (control) and UPF (user) as separate cloud-native functions; CP/UP are "
            "fully decoupled (CUPS), enabling local-breakout and edge compute."
        )


# ═══════════════════════════════════════════════════════════════════════
# Page: CDMA / WCDMA
# ═══════════════════════════════════════════════════════════════════════
elif page == "📻 CDMA / WCDMA":
    from cdma_wcdma import (
        generate_walsh_codes,
        spreading_factor_to_chips,
        despread,
        simulate_cdma_multiuser,
        simulate_near_far,
        simulate_rake_receiver,
    )

    st.title("📻 3G CDMA / WCDMA")
    st.markdown(
        "In CDMA, every user shares the **same frequency at the same time** but uses "
        "a different orthogonal code to spread their data over a wider bandwidth. "
        "Explore the four ideas that made it work: spreading, multi-user separation, "
        "power control, and the RAKE receiver."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧩 Spreading", "👥 Multi-User", "📶 Near-Far + Power Control", "🎚️ RAKE Receiver"]
    )

    # ── Tab 1: Spreading ───────────────────────────────────────────
    with tab1:
        st.markdown(
            "Each data bit is multiplied by a length-`SF` chip code. The chip rate is "
            "`SF × bit rate`, so the signal occupies `SF` times the bandwidth — but at "
            "the receiver, despreading concentrates the signal energy back into one bit "
            "while spreading any narrow-band interference. That's the **processing gain**."
        )
        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            sf_pow = st.slider("Spreading factor (SF = 2^k)", 1, 6, 3, key="cdma_sf_pow")
            sf     = 2 ** sf_pow
            code_idx = st.slider("Walsh code index", 0, sf - 1, min(1, sf - 1),
                                  key="cdma_code_idx")
            bit_val = st.radio("Data bit", [+1, -1], index=0, horizontal=True,
                               key="cdma_bit")
            snr_db_chip = st.slider("Chip-level SNR (dB)", -10, 30, 10,
                                     key="cdma_snr_chip")
            st.metric("Chips per bit", sf)
            st.metric("Processing gain", f"{10 * np.log10(sf):.1f} dB")

        codes = generate_walsh_codes(sf)
        code  = codes[code_idx]
        chips = spreading_factor_to_chips(np.array([bit_val]), code)

        rng_l = np.random.default_rng(0)
        sigp  = float(np.mean(chips ** 2))
        nstd  = np.sqrt(sigp / (10 ** (snr_db_chip / 10)))
        rx    = chips + rng_l.normal(0, nstd, size=chips.shape)
        recovered = float(np.sign(np.dot(rx, code)))

        with col_plot:
            fig, axes = plt.subplots(3, 1, figsize=(10, 6),
                                     gridspec_kw={"hspace": 0.5})

            axes[0].step(range(sf + 1), np.append(code, code[-1]),
                         where="post", color="seagreen", lw=2)
            axes[0].set(title=f"Spreading code (Walsh row {code_idx}, SF={sf})",
                        xlabel="chip", ylabel="±1", yticks=[-1, 0, 1])
            axes[0].grid(True, alpha=0.3)

            axes[1].step(range(sf + 1), np.append(chips, chips[-1]),
                         where="post", color="steelblue", lw=2,
                         label="transmitted chips")
            axes[1].axhline(bit_val, color="tomato", ls="--",
                            label=f"data bit = {bit_val:+d}")
            axes[1].set(title="Transmitted chip stream  (1 bit × code)",
                        xlabel="chip", ylabel="±1", yticks=[-1, 0, 1])
            axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

            axes[2].step(range(sf + 1), np.append(rx, rx[-1]),
                         where="post", color="darkorchid", lw=1.5)
            axes[2].axhline(0, color="k", lw=0.5)
            axes[2].set(title=f"Received (with noise)  →  despread bit = {recovered:+.0f}",
                        xlabel="chip", ylabel="amplitude")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if recovered == bit_val:
            st.success(f"✅ Bit recovered correctly: sent {bit_val:+d}, "
                       f"received {recovered:+.0f}")
        else:
            st.error(f"❌ Bit error — noise overwhelmed the despreader. "
                     f"Lower noise or raise SF.")

    # ── Tab 2: Multi-user CDMA ─────────────────────────────────────
    with tab2:
        st.markdown(
            "Multiple users share the **same channel** at the same time, each using a "
            "different orthogonal Walsh code. The base station despreads with each "
            "user's code in turn — orthogonality makes other users average to zero."
        )

        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            n_users_pow = st.slider("Number of users (SF = users)",
                                     1, 5, 2, key="cdma_mu_users")
            n_users  = 2 ** n_users_pow
            n_bits   = st.slider("Bits per user", 8, 200, 64,
                                  key="cdma_mu_bits")
            snr_db_mu = st.slider("Chip-level SNR (dB)", -5, 30, 15,
                                   key="cdma_mu_snr")

        data, recovered, ber, codes = simulate_cdma_multiuser(
            n_users=n_users, n_bits=n_bits, snr_db=snr_db_mu,
        )

        with col_ctrl:
            st.markdown("---")
            st.markdown("**BER per user**")
            for u, b in ber.items():
                st.metric(f"User {u}", f"{b:.4f}")

        with col_plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                                     gridspec_kw={"hspace": 0.5})

            colors = plt.cm.tab10(np.linspace(0, 1, max(n_users, 4)))
            for u in range(n_users):
                offset = u * 0.18
                axes[0].scatter(range(n_bits), data[u] + offset,
                                marker="s", s=18, color=colors[u],
                                label=f"sent  U{u}")
                axes[0].scatter(range(n_bits), recovered[u] + offset,
                                marker="x", s=22, color=colors[u], alpha=0.6)
            axes[0].set(title=f"{n_users} users sharing the same channel "
                              f"(■ sent, × recovered)",
                        xlabel="bit index", ylabel="±1 (offset per user)")
            axes[0].legend(fontsize=8, loc="upper right", ncol=2)
            axes[0].grid(True, alpha=0.3)

            # Cross-correlation matrix
            xc = (codes @ codes.T) / codes.shape[1]
            im = axes[1].imshow(xc, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            axes[1].set(title=f"Walsh code cross-correlation matrix "
                              f"(diagonal=1, off-diagonal=0 → orthogonal)",
                        xlabel="code j", ylabel="code i")
            plt.colorbar(im, ax=axes[1], fraction=0.04)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "**Why it works:** because the Walsh codes are orthogonal "
            "(`<c_i, c_j> = 0` for `i ≠ j`), correlating the received sum with code `i` "
            "preserves user `i` and cancels every other user. The off-diagonal of the "
            "cross-correlation matrix is exactly zero."
        )

    # ── Tab 3: Near-Far + Power Control ────────────────────────────
    with tab3:
        st.markdown(
            "Path loss grows roughly as `d^3.5` in urban environments. Without power "
            "control, the close user **drowns out** the far one — orthogonality alone "
            "can't save you. WCDMA's TPC inner loop runs **1500 times per second**, "
            "telling each UE to step its TX power up or down by 1 dB to land at the "
            "same Rx power at the Node B."
        )

        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            d1 = st.slider("User 1 distance (m)",   50, 3000, 100,  step=50, key="cdma_d1")
            d2 = st.slider("User 2 distance (m)",   50, 3000, 500,  step=50, key="cdma_d2")
            d3 = st.slider("User 3 distance (m)",   50, 3000, 1000, step=50, key="cdma_d3")
            d4 = st.slider("User 4 distance (m)",   50, 3000, 2000, step=50, key="cdma_d4")

        distances = [d1, d2, d3, d4]
        rx_no_ctrl, rx_history, target = simulate_near_far(distances)

        with col_ctrl:
            st.markdown("---")
            st.metric("Spread without TPC",
                      f"{rx_no_ctrl.max() - rx_no_ctrl.min():.1f} dB")
            converged = np.argmin(np.abs(rx_history[:, -1] - target))
            st.metric("TPC convergence", f"~{converged} steps")

        with col_plot:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                                     gridspec_kw={"wspace": 0.35})

            axes[0].bar(range(1, 5), rx_no_ctrl,
                        color=["steelblue", "tomato", "seagreen", "darkorchid"])
            axes[0].axhline(target, color="k", ls="--",
                            label=f"target {target:+.0f} dBm")
            axes[0].set(title="Without power control",
                        xlabel="user", ylabel="Rx power at Node B (dBm)",
                        xticks=range(1, 5))
            for i, v in enumerate(rx_no_ctrl):
                axes[0].text(i + 1, v + 1, f"{v:+.0f}", ha="center", fontsize=9)
            axes[0].legend(fontsize=9); axes[0].grid(True, axis="y", alpha=0.3)

            cs = ["steelblue", "tomato", "seagreen", "darkorchid"]
            for u, d in enumerate(distances):
                axes[1].plot(rx_history[:, u], color=cs[u], lw=1.5,
                             label=f"U{u+1} ({d} m)")
            axes[1].axhline(target, color="k", ls="--", label="target")
            axes[1].set(title="With closed-loop TPC",
                        xlabel="TPC iteration  (1500 / s)",
                        ylabel="Rx power at Node B (dBm)")
            axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.success(
            "**The point:** without TPC, even small distance differences create "
            "30+ dB Rx-power gaps, swamping the orthogonality budget. After TPC, "
            "every user lands at the same target power — only then can the despreader "
            "actually separate them."
        )

    # ── Tab 4: RAKE Receiver ───────────────────────────────────────
    with tab4:
        st.markdown(
            "A WCDMA chip period is ~0.26 µs (chip rate = 3.84 Mcps). Echoes that "
            "arrive more than one chip late are *resolvable* — the RAKE receiver puts "
            "one matched-filter **finger** on each echo and combines them with "
            "**Maximum Ratio Combining**, turning multipath from a problem into "
            "diversity gain."
        )

        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            sf_r   = st.select_slider("Spreading factor",
                                       options=[4, 8, 16, 32, 64], value=16,
                                       key="cdma_rake_sf")
            n_b    = st.slider("Bits per trial", 50, 500, 200, step=50,
                                key="cdma_rake_bits")
            snr_pt = st.slider("Show single-shot at SNR (dB)", -10, 20, 0,
                                key="cdma_rake_snr_pt")

        # single-shot
        ber_s_pt, ber_r_pt, paths = simulate_rake_receiver(
            sf=sf_r, n_data_bits=n_b, snr_db=snr_pt,
        )
        with col_ctrl:
            st.markdown("---")
            st.metric("Single finger BER",      f"{ber_s_pt:.4f}")
            st.metric("RAKE (3 fingers) BER",   f"{ber_r_pt:.4f}")

        # BER vs SNR sweep (cached)
        @st.cache_data(show_spinner="Sweeping SNR...")
        def _rake_sweep(sf, n_b):
            snrs = list(range(-10, 16, 2))
            single, rake = [], []
            for s in snrs:
                bs, br, _ = simulate_rake_receiver(
                    sf=sf, n_data_bits=n_b, snr_db=s,
                )
                single.append(bs)
                rake.append(br)
            return snrs, single, rake

        snrs, ber_s, ber_r = _rake_sweep(sf_r, n_b)

        with col_plot:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                                     gridspec_kw={"wspace": 0.35})

            d_axis = [d for d, _ in paths]
            a_axis = [a for _, a in paths]
            ml, sl, _ = axes[0].stem(d_axis, a_axis, basefmt="k-")
            plt.setp(sl, color="steelblue", linewidth=2)
            plt.setp(ml, color="steelblue", markersize=10)
            for d, a in zip(d_axis, a_axis):
                axes[0].annotate(f"{a:.2f}", xy=(d, a),
                                 xytext=(d + 0.15, a + 0.04), fontsize=9)
            axes[0].set(title="Multipath channel (3 paths)",
                        xlabel="delay (chips)", ylabel="amplitude",
                        xlim=[-0.5, max(d_axis) + 1.5], ylim=[0, 1.15])
            axes[0].grid(True, alpha=0.3)

            axes[1].semilogy(snrs, np.clip(ber_s, 1e-4, 1),
                             "tomato", marker="o", lw=2, label="single finger")
            axes[1].semilogy(snrs, np.clip(ber_r, 1e-4, 1),
                             "steelblue", marker="s", lw=2,
                             label="RAKE (3 fingers, MRC)")
            axes[1].axvline(snr_pt, color="k", ls=":", alpha=0.6,
                            label=f"current SNR = {snr_pt} dB")
            axes[1].set(title="BER vs SNR — RAKE diversity gain",
                        xlabel="SNR (dB)", ylabel="BER",
                        ylim=[1e-4, 1])
            axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "**MRC weighting:** each finger is weighted by the path's amplitude before "
            "summing — the strong direct path counts more than weak echoes, which is "
            "the optimal way to combine independent noisy copies of the same signal."
        )
