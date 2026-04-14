# Signal Fundamentals

## Topics
- Continuous vs discrete signals
- Fourier Transform & DFT
- Nyquist sampling theorem
- Signal power, energy, bandwidth
- Noise: thermal noise, SNR

## Files — Topic 2: RF & Wireless Basics

| File | What it demonstrates |
|------|----------------------|
| [rf_basics.py](rf_basics.py) | Frequency ↔ wavelength, dB/dBm conversions, Shannon capacity |
| [path_loss.py](path_loss.py) | FSPL, log-distance model, log-normal shadowing + plots |
| [link_budget.py](link_budget.py) | Full link budget calculator: EIRP → path loss → margin → PASS/FAIL |

## How to Run

```bash
cd 01_Signal_Fundamentals
pip install matplotlib numpy

python rf_basics.py     # frequency/wavelength table, dB conversions, Shannon capacity
python path_loss.py     # path loss models + saves fspl_vs_frequency.png, path_loss_shadowing.png
python link_budget.py   # 3 link budget scenarios: 3.5 GHz, 28 GHz @ 500m, 28 GHz @ 1km
```

## Key Takeaways
- Higher frequency → shorter wavelength → more spectrum available → wider bandwidth → more capacity
- FSPL increases 6 dB every time you double distance OR double frequency
- PLE ≈ 2 in LOS (regardless of frequency); NLOS mmWave PLE can reach 4–6
- Link margin = Rx Power − Rx Sensitivity; must be > 0 for the link to work
- mmWave needs beamforming (high antenna gain) to compensate for its severe path loss
