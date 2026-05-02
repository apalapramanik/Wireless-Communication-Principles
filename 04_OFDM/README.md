# Topic 5 — OFDM Full Transceiver

OFDM (Orthogonal Frequency Division Multiplexing) is the physical-layer foundation of LTE, 5G NR, Wi-Fi 4/5/6, and most modern broadband systems. This module builds a complete OFDM link from scratch in four parts.

---

## Parameters (5G NR-like)

| Parameter | Value | Notes |
|-----------|-------|-------|
| FFT size `N_FFT` | 128 | Total subcarrier slots |
| Cyclic prefix `N_CP` | 16 | Samples prepended per symbol |
| Active subcarriers `N_ACTIVE` | 72 | Data-bearing; rest are guard bands |
| Modulation `M` | 16-QAM | 4 bits per symbol |
| Channel model | 3-tap urban macro | Delays: 0, 4, 9 samples |

---

## Part A — QAM Mapper + OFDM Modulator / Demodulator

### How it works

**QAM modulation** groups the bit stream into chunks of `log2(M)` bits and maps each chunk to a complex symbol on the constellation grid. For 16-QAM, 4 bits → one of 16 points on a 4×4 grid. The constellation is power-normalized so `E[|s|²] = 1`.

**OFDM modulation** takes `N_ACTIVE` QAM symbols and spreads them across subcarriers in the frequency domain:
1. Place symbols on active subcarrier bins (DC-centred, guard bands left empty at edges)
2. **IFFT** converts the frequency-domain frame to a time-domain waveform
3. Copy the last `N_CP` samples to the front — this is the **cyclic prefix (CP)**

The CP is the key OFDM trick. Any multipath channel with delay ≤ N_CP turns linear convolution into circular convolution, which becomes a simple per-subcarrier multiplication in the frequency domain after the FFT at the receiver.

**OFDM demodulation** reverses the process: strip CP → FFT → extract the same subcarrier bins.

### Sanity check output (no channel, no noise)

```
PART A — Sanity check (no channel, no noise)
Tx symbol power : 1.0333
Rx symbol power : 1.0333
Max IQ error    : 3.72e-16
Expected: max error < 1e-13 (machine precision)
```

Max IQ error of 3.72e-16 confirms the IFFT/FFT pair is a perfect round-trip at machine precision — no bugs in the modulator/demodulator before adding any channel impairments.

---

## Part B — Multipath Channel Model

### How it works

The channel is a **tapped delay line** — the received signal is a sum of scaled, delayed copies of the transmitted signal:

```
y[n] = Σ g_k · x[n - d_k] + noise
```

Three paths model an urban macro cell:

| Path | Delay (samples) | Complex gain | Relative power |
|------|-----------------|--------------|----------------|
| LOS  | 0  | 1.0 | 0 dB |
| Reflection 1 | 4 | 0.6 e^{j0.8} | −4.4 dB |
| Reflection 2 | 9 | 0.3 e^{-j1.2} | −10.5 dB |

The max delay is 9 samples. The CP length is 16 samples, so **CP > max delay** → the CP fully absorbs all inter-symbol interference (ISI).

### Channel plots

![Channel Model](channel.png)

**Left — Impulse response:** Three spikes at samples 0, 4, 9. The red dashed line at sample 16 shows the CP boundary — all energy arrives well within the CP window.

**Right — Frequency response:** The three paths interfere constructively and destructively at different frequencies, creating a **frequency-selective fading** pattern (some subcarriers are attenuated by up to ~6 dB, others boosted). This is exactly what the equalizer in Part C must correct.

---

## Part C — Full Link with Zero-Forcing Equalizer

### How it works

With the CP in place, each OFDM subcarrier `k` sees an independent scalar channel:

```
Y_k = H_k · X_k + N_k
```

where `H_k` is the channel's frequency response at subcarrier `k` (just one complex number per subcarrier). This is why OFDM is so powerful — a wideband multipath channel turns into many independent flat-fading narrowband channels.

**Zero-Forcing (ZF) equalization** inverts this scalar per-subcarrier:

```
X̂_k = Y_k / H_k
```

The equalizer assumes perfect channel knowledge (`H_k` is precomputed from the known channel taps). In a real system, `H_k` would be estimated from pilot subcarriers.

### Link simulation results (SNR = 25 dB, 50 OFDM symbols)

```
BER with ZF equalizer  : 0.0045
BER without equalizer  : 0.2115
```

A ~47× reduction in BER. Without equalization, every subcarrier lands at a different wrong position because each sees a different `H_k` rotation and scaling — this is not random noise, it is deterministic channel distortion.

### Constellation plots

![Constellation](ofdm_constellation.png)

**Left — Ideal 16-QAM:** The 16 reference points at clean grid positions.

**Center — No equalizer:** The 16 clusters are smeared and displaced in all directions. Each subcarrier's `H_k` scales and rotates its symbols differently, so the received cloud has no consistent structure. Adding more transmit power does not fix this — the distortion is deterministic, not noise-driven.

**Right — ZF equalizer:** The 16 clusters snap back to the correct positions. The residual spread around each point is pure AWGN noise — the channel distortion is fully undone. Visible spread is expected at 25 dB SNR for 16-QAM.

---

## Part D — BER vs SNR Curve

### How it works

The theoretical BER for 16-QAM in AWGN is:

```
BER = (3/8) · erfc(√(SNR/10))
```

This is the best-case floor assuming no multipath — a perfect AWGN channel.

### Results

| SNR (dB) | BER — ZF equalizer | BER — no equalizer |
|----------|--------------------|--------------------|
| 0  | 2.81e-01 | 3.21e-01 |
| 6  | 1.51e-01 | 2.56e-01 |
| 12 | 4.83e-02 | 2.21e-01 |
| 18 | 1.69e-02 | 2.05e-01 |
| 24 | 3.82e-03 | 2.05e-01 |
| 27 | 9.26e-04 | 2.05e-01 |
| 30 | 1.16e-04 | 2.03e-01 |

### BER curve

![BER vs SNR](ofdm_ber.png)

**What the curves show:**

- **No equalizer (red dashed):** BER floors at ~0.20 for all SNR above ~15 dB. Adding power stops helping entirely. The channel is deterministically distorting every subcarrier — this is an irreducible error floor caused by multipath, not noise.

- **ZF equalizer (blue solid):** BER falls steadily with SNR, tracking close to the theoretical AWGN curve. This confirms that CP + FFT + one-tap ZF division fully neutralizes the 3-tap multipath channel.

- **Small gap from theory:** The ZF equalizer slightly amplifies noise on subcarriers where `|H_k|` is small (this is the ZF noise enhancement penalty). An MMSE equalizer would reduce this gap at low SNR.

- **10⁻³ BER target:** The ZF equalizer crosses the 1e-3 line at approximately **27 dB SNR**, close to the theoretical requirement for 16-QAM.

---

## Running

```bash
python ofdm_transceiver.py
```

Outputs: `channel.png`, `ofdm_constellation.png`, `ofdm_ber.png`

Dependencies: `numpy`, `matplotlib`, `scipy`

---

## Key Concepts Summary

| Concept | What it does |
|---------|-------------|
| IFFT at TX | Converts frequency-domain QAM symbols to time-domain waveform |
| Cyclic Prefix | Converts linear convolution (multipath) into circular — enables one-tap equalization |
| FFT at RX | Converts received waveform back to frequency domain |
| ZF Equalizer | Divides each subcarrier by its channel coefficient — undoes per-subcarrier rotation/scaling |
| Frequency selectivity | Different subcarriers see different channel gains — some attenuated, some boosted |
| ZF noise enhancement | Subcarriers with low `\|H_k\|` get amplified noise when dividing — MMSE avoids this |
