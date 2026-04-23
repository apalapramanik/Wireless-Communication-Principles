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

# Allow importing from topic folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '02_Modulation_Techniques'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_Signal_Fundamentals'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '03_DSP'))

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
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("All implementations from scratch — no scipy")


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
# Page: Overview
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Wireless Communication Principles")
    st.markdown(
        "An interactive companion to the repo — explore every core concept with live controls."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Topics covered", "5")
    col2.metric("Python files", "16+")
    col3.metric("External deps", "numpy + matplotlib")

    st.markdown("---")
    st.markdown("""
| Page | What you can explore |
|------|----------------------|
| 📡 **FFT Explorer** | Build any composite signal, see its spectrum live |
| ⚡ **Aliasing Demo** | Slide the sampling rate below Nyquist and watch aliasing happen |
| 🗺️ **Constellation Viewer** | Pick modulation order, add noise, see symbols scatter |
| 📉 **BER Curves** | Theory vs simulation — how SNR determines error rate |
| 📶 **Path Loss & Link Budget** | Tune distance, frequency, antenna gains — PASS or FAIL |
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
