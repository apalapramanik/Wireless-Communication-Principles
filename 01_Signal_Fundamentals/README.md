# Topic 2 — Signal Fundamentals & RF Basics

Source: Rappaport Ch. 1–3 | Chat discussion

## Topics
- RF fundamentals: frequency, wavelength, dB/dBm
- Path loss models: FSPL, log-distance, log-normal shadowing
- Link budget: EIRP → path loss → receiver sensitivity → margin

## Files

| File | What it demonstrates |
|------|----------------------|
| [rf_basics.py](rf_basics.py) | Frequency ↔ wavelength, dB/dBm conversions, Shannon capacity |
| [path_loss.py](path_loss.py) | FSPL, log-distance model, log-normal shadowing + plots |
| [link_budget.py](link_budget.py) | Full link budget calculator: EIRP → path loss → margin → PASS/FAIL |

## How to Run

```bash
cd 01_Signal_Fundamentals
pip install matplotlib numpy

python3 rf_basics.py     # frequency/wavelength table, dB conversions, Shannon capacity
python3 path_loss.py     # path loss models + saves fspl_vs_frequency.png, path_loss_shadowing.png
python3 link_budget.py   # 3 link budget scenarios: 3.5 GHz, 28 GHz @ 200m LOS, 28 GHz @ 300m NLOS
```

## Plots

![FSPL vs Distance](fspl_vs_frequency.png)

![Path Loss with Shadowing](path_loss_shadowing.png)

## Sample Outputs

**`rf_basics.py`**
```
=== Frequency → Wavelength → Band ===

  Band                              λ (cm)  Category
  700 MHz  (4G low band)       42.86  Sub-3 GHz   (4G / WiFi 2.4 GHz territory)
  2.1 GHz  (4G mid band)       14.29  Sub-3 GHz   (4G / WiFi 2.4 GHz territory)
  3.5 GHz  (5G NR FR1)          8.57  Sub-6 GHz   (5G NR FR1)
  28 GHz   (5G mmWave)          1.07  cmWave      (5G NR FR1 upper / satellite)
  39 GHz   (5G mmWave)          0.77  mmWave      (5G NR FR2 — 28 / 39 GHz)

=== Shannon Capacity: Bandwidth → Data Rate ===
  4G  20 MHz   SNR=20 dB        C =    664.4 Mbps
  5G  100 MHz  SNR=20 dB        C =   3321.9 Mbps
  mmWave 400 MHz SNR=30dB       C = 132877.1 Mbps
```

**`path_loss.py`**
```
  1 km, 700 MHz (4G)    FSPL =  89.4 dB
  1 km, 3.5 GHz (5G)    FSPL = 103.3 dB
  1 km, 28 GHz (mmWave) FSPL = 121.4 dB
```

**`link_budget.py`**
```
  5G NR — 3.5 GHz  |  500 m  |  urban NLOS   →  Margin = +24.2 dB  PASS ✅
  5G mmWave — 28 GHz  |  200 m  |  LOS        →  Margin = +35.6 dB  PASS ✅
  5G mmWave — 28 GHz  |  300 m  |  NLOS       →  Margin =  -9.3 dB  FAIL ❌
```

## Key Takeaways
- Higher frequency → shorter wavelength → more spectrum available → wider bandwidth → more capacity
- FSPL increases 6 dB every time you double distance OR double frequency
- PLE ≈ 2 in LOS (regardless of frequency); NLOS mmWave PLE can reach 4–6
- Link margin = Rx Power − Rx Sensitivity; must be > 0 for the link to work
- mmWave needs beamforming (high antenna gain) to compensate for its severe path loss
