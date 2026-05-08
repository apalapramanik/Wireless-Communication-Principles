# Wireless Communication Principles

A topic-wise organized reference and implementation repo for core wireless communication concepts — from networking fundamentals through 5G NR.

---

## Interactive Dashboard

```bash
pip install streamlit numpy matplotlib scipy
streamlit run app.py
```

![Dashboard demo](screenshots/demo.gif)

| Page | What you can do |
|------|-----------------|
| 🏠 Overview | Summary of all pages |
| 📡 FFT Explorer | Build a composite signal from up to 3 tones, see live spectrum |
| ⚡ Aliasing Demo | Drag sampling rate below Nyquist and watch aliasing happen |
| 🗺️ Constellation Viewer | Pick modulation order, add AWGN noise, watch symbols scatter |
| 📉 BER Curves | Theory vs Monte Carlo — how SNR drives error rate |
| 📶 Path Loss & Link Budget | Tune distance / frequency / antenna gains — PASS or FAIL |
| 🔀 OFDM Explorer | Build a multipath channel, equalize it, measure BER vs SNR |
| 🛰️ Mobile Network Architecture | Walk through the 4G LTE attach procedure, message by message |

### Dashboard screenshots

| FFT Explorer | Aliasing Demo |
|---|---|
| ![FFT](screenshots/fft_explorer.png) | ![Aliasing](screenshots/aliasing_demo.png) |

| Constellation Viewer | BER Curves |
|---|---|
| ![Constellation](screenshots/constellation_viewer.png) | ![BER](screenshots/ber_curves.png) |

| Path Loss & Link Budget | OFDM Explorer |
|---|---|
| ![Path Loss](screenshots/path_loss.png) | ![OFDM](screenshots/ofdm_explorer.png) |

| Mobile Network Architecture | |
|---|---|
| ![Mobile Network Architecture](screenshots/mobile_network_architecture.png) | |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.10+ | All implementations |
| Numerics | NumPy | Signal processing, matrix ops, FFT |
| Plotting | Matplotlib | All static plots and figures |
| Dashboard | Streamlit | Interactive visualizations (`app.py`) |
| DSP | Pure NumPy | FIR design, STFT, OFDM modulator — built from scratch |
| Theory | SciPy | `erfc` for theoretical BER reference curves |
| Version control | Git + GitHub | This repo |

All signal processing is built from scratch with NumPy — no black-box DSP functions. SciPy is used only for the `erfc` function in theoretical BER formulas.

---

## Structure

### Phase 1 — Physical Layer & Signal Processing

| Folder | Topic | Key files |
|--------|-------|-----------|
| [00_Networking_Fundamentals](00_Networking_Fundamentals/) | TCP/UDP, packet switching, 4 delays, encapsulation | `delay_calculator.py` `tcp_demo.py` `udp_demo.py` |
| [01_Signal_Fundamentals](01_Signal_Fundamentals/) | RF basics, dB/dBm, path loss models, link budget | `rf_basics.py` `path_loss.py` `link_budget.py` |
| [02_Modulation_Techniques](02_Modulation_Techniques/) | QAM constellations, BER curves, 5G MCS table | `constellation.py` `transceiver.py` `ber_curves.py` `mcs_table.py` |
| [03_DSP](03_DSP/) | FFT, Nyquist/aliasing, FIR filter, spectrogram | `fft_basics.py` `aliasing.py` `fir_filter.py` `spectrogram.py` |
| [04_OFDM](04_OFDM/) | Full transceiver, multipath channel, ZF equalizer, BER vs SNR | `ofdm_transceiver.py` |

### Phase 2 — Mobile Network Architecture & Protocols

| Folder | Topic | Key files |
|--------|-------|-----------|
| [05_Mobile_Network_Architecture](05_Mobile_Network_Architecture/) | 4G LTE attach procedure, control vs user plane, GTP tunnels, 2G→5G evolution | `lte_attach.py` |

*More Phase 2 topics will be added as the series progresses.*

Each folder has its own deep-dive README with theory, simulation results, and figures — see [04_OFDM/README.md](04_OFDM/README.md) and [05_Mobile_Network_Architecture/README.md](05_Mobile_Network_Architecture/README.md).

---

## Goals
- Implement every concept from scratch — no black-box libraries
- Connect theory (formulas) to simulation (code) to visualization (plots)
- Build intuition for system-level 5G design
