"""
Topic 2 — RF & Wireless Basics
Full RF link budget calculator.

Received Power = EIRP - Path Loss + Rx Antenna Gain - Misc Losses
Link Margin    = Rx Power - Rx Sensitivity
                 > 0  →  PASS   (link works)
                 < 0  →  FAIL   (too far, or too noisy)

Rx Sensitivity (dBm) = Thermal Noise Floor + Noise Figure + Required SNR
Thermal Noise Floor  = -174 + 10·log10(BW_Hz)   [at 290 K]

Reference distance d0 = 100 m (standard for cellular models).
Path loss at d0 is taken as FSPL(d0), then the PLE governs decay beyond that.

Run:
    python link_budget.py
"""

import numpy as np
from path_loss import fspl_db, log_distance_pl


def link_budget(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    freq_hz: float,
    distance_m: float,
    bandwidth_hz: float,
    noise_figure_db: float,
    required_snr_db: float,
    misc_losses_db: float = 0.0,
    path_loss_exponent: float = 2.0,
    d0_m: float = 100.0,
) -> dict:
    """
    Compute a complete RF link budget.

    Args:
        tx_power_dbm      : base station / device transmit power (dBm)
        tx_gain_dbi       : Tx antenna gain (dBi)
        rx_gain_dbi       : Rx antenna gain (dBi)
        freq_hz           : carrier frequency (Hz)
        distance_m        : link distance (m)
        bandwidth_hz      : channel bandwidth (Hz)
        noise_figure_db   : receiver noise figure
        required_snr_db   : minimum SNR for target modulation (e.g. 10 dB for 64-QAM)
        misc_losses_db    : cable, connector, body, penetration losses
        path_loss_exponent: n=2 free space; 3–4 urban NLOS; 4–6 indoors
        d0_m              : reference distance (100 m is standard for cellular)

    Returns:
        dict with all intermediate values and PASS / FAIL result.
    """
    eirp_dbm = tx_power_dbm + tx_gain_dbi

    if path_loss_exponent == 2.0:
        pl_db = fspl_db(distance_m, freq_hz)
    else:
        pl_d0 = fspl_db(d0_m, freq_hz)
        pl_db = log_distance_pl(distance_m, d0_m, pl_d0, path_loss_exponent)

    rx_power_dbm        = eirp_dbm - pl_db + rx_gain_dbi - misc_losses_db
    thermal_noise_dbm   = -174 + 10 * np.log10(bandwidth_hz)
    rx_sensitivity_dbm  = thermal_noise_dbm + noise_figure_db + required_snr_db
    margin_db           = rx_power_dbm - rx_sensitivity_dbm

    return {
        "Tx Power (dBm)":           tx_power_dbm,
        "Tx Antenna Gain (dBi)":    tx_gain_dbi,
        "EIRP (dBm)":               eirp_dbm,
        "Path Loss (dB)":           round(pl_db, 2),
        "Rx Antenna Gain (dBi)":    rx_gain_dbi,
        "Misc Losses (dB)":         misc_losses_db,
        "Rx Power (dBm)":           round(rx_power_dbm, 2),
        "──────────────────────":   "─────────────────",
        "Thermal Noise Floor(dBm)": round(thermal_noise_dbm, 2),
        "Noise Figure (dB)":        noise_figure_db,
        "Required SNR (dB)":        required_snr_db,
        "Rx Sensitivity (dBm)":     round(rx_sensitivity_dbm, 2),
        "──────────────────────2":  "─────────────────",
        "Link Margin (dB)":         round(margin_db, 2),
        "Result":                   "PASS ✅" if margin_db > 0 else "FAIL ❌",
    }


def print_budget(title: str, result: dict) -> None:
    width = 54
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    for k, v in result.items():
        if k.startswith("──"):
            print("  " + "-" * (width - 2))
        else:
            print(f"  {k:<30} {v}")
    print()


if __name__ == "__main__":
    # Scenario A — 5G gNB at 3.5 GHz, 500 m, urban NLOS
    result_a = link_budget(
        tx_power_dbm       = 43,       # 20 W gNB
        tx_gain_dbi        = 15,       # sector antenna
        rx_gain_dbi        = 0,        # omnidirectional UE
        freq_hz            = 3.5e9,
        distance_m         = 500,
        bandwidth_hz       = 100e6,    # 100 MHz NR channel
        noise_figure_db    = 7,        # UE noise figure
        required_snr_db    = 10,       # 64-QAM threshold
        misc_losses_db     = 3,        # body + cable
        path_loss_exponent = 3.5,      # urban NLOS
        d0_m               = 100,
    )
    print_budget("5G NR — 3.5 GHz  |  500 m  |  urban NLOS", result_a)

    # Scenario B — 5G mmWave at 28 GHz, 200 m LOS (realistic mmWave deployment)
    result_b = link_budget(
        tx_power_dbm       = 43,
        tx_gain_dbi        = 25,       # beamforming gain
        rx_gain_dbi        = 10,       # UE phased array
        freq_hz            = 28e9,
        distance_m         = 200,
        bandwidth_hz       = 400e6,    # 400 MHz mmWave channel
        noise_figure_db    = 10,
        required_snr_db    = 10,
        misc_losses_db     = 3,
        path_loss_exponent = 2.0,      # LOS (mmWave is deployed LOS only)
    )
    print_budget("5G NR mmWave — 28 GHz  |  200 m  |  LOS", result_b)

    # Scenario C — mmWave 28 GHz, outdoor-to-indoor (wall penetration kills the link)
    # Wall penetration loss at 28 GHz is 20–40 dB (concrete ~30 dB, glass ~10 dB)
    # Beamforming gain also degrades in deep NLOS; model as reduced Rx gain
    result_c = link_budget(
        tx_power_dbm       = 43,
        tx_gain_dbi        = 25,
        rx_gain_dbi        = 5,        # reduced — beam can't steer through a wall
        freq_hz            = 28e9,
        distance_m         = 300,
        bandwidth_hz       = 400e6,
        noise_figure_db    = 10,
        required_snr_db    = 10,
        misc_losses_db     = 25,       # ~20 dB wall penetration + 5 dB cable/body
        path_loss_exponent = 5.0,      # dense NLOS + obstructions
        d0_m               = 100,
    )
    print_budget("5G NR mmWave — 28 GHz  |  300 m  |  outdoor-to-indoor NLOS", result_c)

    print("Observations:")
    print("  • 3.5 GHz at 500 m passes comfortably — wide coverage, the workhorse of 5G.")
    print("  • 28 GHz at 200 m in LOS passes with a huge margin — gigabit speeds nearby.")
    print("  • 28 GHz outdoor-to-indoor fails — a single concrete wall adds ~20 dB,")
    print("    which exhausts the link margin entirely at 300 m.")
    print("  • This is why mmWave is only deployed outdoors in dense small-cell clusters.")
