"""
Phase 2 | Topic 1: Mobile Network Architecture Fundamentals
Simulator: 4G LTE Attach Procedure and End-to-End Data Flow

This script models the key message exchanges that happen when your phone
connects to a 4G LTE network and establishes a data session.

Domains simulated:
    UE -> eNB -> MME (control plane)
    UE -> eNB -> SGW -> PGW (user plane)
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Network nodes
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    name: str
    domain: str  # "ue", "ran", "core_cp", "core_up"

    def __str__(self):
        return self.name


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

@dataclass
class Message:
    src: NetworkNode
    dst: NetworkNode
    name: str
    plane: str          # "control" or "user"
    payload: dict = field(default_factory=dict)
    interface: str = ""

    def display(self, indent: int = 0):
        arrow = "-->" if self.plane == "control" else "==>"
        plane_tag = "[CP]" if self.plane == "control" else "[UP]"
        iface = f" ({self.interface})" if self.interface else ""
        prefix = "  " * indent
        print(f"{prefix}{plane_tag} {self.src.name} {arrow} {self.dst.name}{iface}")
        print(f"{prefix}    Message : {self.name}")
        if self.payload:
            for k, v in self.payload.items():
                print(f"{prefix}    {k:18s}: {v}")


# ---------------------------------------------------------------------------
# Procedure engine
# ---------------------------------------------------------------------------

class ProcedureEngine:
    """
    Replays a sequence of messages and prints an annotated trace.
    Tracks state across the network as the procedure progresses.
    """

    def __init__(self):
        # Define the network nodes
        self.ue     = NetworkNode("UE",    "ue")
        self.enb    = NetworkNode("eNB",   "ran")
        self.mme    = NetworkNode("MME",   "core_cp")
        self.hss    = NetworkNode("HSS",   "core_cp")
        self.sgw    = NetworkNode("SGW",   "core_up")
        self.pgw    = NetworkNode("PGW",   "core_up")

        self.state: dict = {}  # shared network state built up as messages flow

    def run_step(self, msg: Message, notes: Optional[str] = None, delay: float = 0.05):
        msg.display(indent=1)
        if notes:
            print(f"    *** {notes}")
        self.state.update(msg.payload)
        print()
        time.sleep(delay)

    def section(self, title: str):
        width = 60
        print("=" * width)
        print(f"  {title}")
        print("=" * width)

    def subsection(self, title: str):
        print(f"\n  --- {title} ---")


# ---------------------------------------------------------------------------
# LTE attach procedure
# ---------------------------------------------------------------------------

def run_lte_attach(engine: ProcedureEngine):
    e = engine
    ue, enb, mme, hss, sgw, pgw = e.ue, e.enb, e.mme, e.hss, e.sgw, e.pgw

    e.section("4G LTE: UE Attach and Default Bearer Setup")

    # ------------------------------------------------------------------ #
    # Step 1: RRC Connection (UE <-> eNB)
    # ------------------------------------------------------------------ #
    e.subsection("Phase 1: RRC Connection Setup (UE <-> eNB, Uu interface)")

    e.run_step(Message(ue, enb, "RRC Connection Request", "control",
        {"cause": "mo-Data", "ue_category": "Cat 4"},
        interface="Uu"),
        notes="UE requests radio resources from eNB")

    e.run_step(Message(enb, ue, "RRC Connection Setup", "control",
        {"srb1_config": "granted", "initial_ul_grant": "6 PRBs"},
        interface="Uu"),
        notes="eNB allocates a Signalling Radio Bearer (SRB1)")

    e.run_step(Message(ue, enb, "RRC Connection Setup Complete", "control",
        {"nas_message": "Attach Request (piggybacked)"},
        interface="Uu"),
        notes="UE confirms SRB1 and piggybacks the NAS Attach Request")

    # ------------------------------------------------------------------ #
    # Step 2: Initial UE Message (eNB -> MME)
    # ------------------------------------------------------------------ #
    e.subsection("Phase 2: Initial UE Message (eNB -> MME, S1-MME interface)")

    e.run_step(Message(enb, mme, "Initial UE Message", "control",
        {"nas_pdu": "Attach Request",
         "tai":     "MCC=001, MNC=01, TAC=0x0001",
         "ecgi":    "Cell ID 0x0001"},
        interface="S1-MME"),
        notes="eNB forwards NAS message to MME; adds location info")

    # ------------------------------------------------------------------ #
    # Step 3: Authentication (MME <-> HSS <-> UE)
    # ------------------------------------------------------------------ #
    e.subsection("Phase 3: Authentication (EPS-AKA)")

    e.run_step(Message(mme, hss, "Authentication Information Request", "control",
        {"imsi": "001010000000001"},
        interface="S6a"),
        notes="MME asks HSS for authentication vectors for this IMSI")

    e.run_step(Message(hss, mme, "Authentication Information Answer", "control",
        {"rand": "0xA1B2C3...", "autn": "0xD4E5F6...", "xres": "0x11223344",
         "kasme": "derived from K and SQN"},
        interface="S6a"),
        notes="HSS returns authentication vectors; xres is the expected response")

    e.run_step(Message(mme, ue, "Authentication Request", "control",
        {"rand": "0xA1B2C3...", "autn": "0xD4E5F6..."},
        interface="NAS (via eNB)"),
        notes="MME sends challenge to UE")

    e.run_step(Message(ue, mme, "Authentication Response", "control",
        {"res": "0x11223344"},
        interface="NAS (via eNB)"),
        notes="UE computes RES using its SIM secret key K; MME verifies RES == xres")

    e.state["authenticated"] = True

    # ------------------------------------------------------------------ #
    # Step 4: Security mode (NAS + AS)
    # ------------------------------------------------------------------ #
    e.subsection("Phase 4: Security Mode Command (ciphering + integrity)")

    e.run_step(Message(mme, ue, "Security Mode Command", "control",
        {"selected_nas_cipher": "AES-128-CTR", "selected_nas_integ": "AES-128-CMAC"},
        interface="NAS"),
        notes="Activates NAS-layer ciphering and integrity protection")

    e.run_step(Message(ue, mme, "Security Mode Complete", "control", {},
        interface="NAS (ciphered)"),
        notes="All subsequent NAS messages are now ciphered and integrity-protected")

    # ------------------------------------------------------------------ #
    # Step 5: Create Session (MME -> SGW -> PGW)
    # ------------------------------------------------------------------ #
    e.subsection("Phase 5: Default EPS Bearer Setup (MME -> SGW -> PGW)")

    e.run_step(Message(mme, sgw, "Create Session Request", "control",
        {"imsi":    "001010000000001",
         "apn":     "internet",
         "pdn_type": "IPv4",
         "bearer_id": "5"},
        interface="S11"),
        notes="MME requests a GTP tunnel from SGW to PGW for this UE")

    e.run_step(Message(sgw, pgw, "Create Session Request", "control",
        {"apn":      "internet",
         "bearer_id": "5",
         "qci":       "9 (best-effort)"},
        interface="S5"),
        notes="SGW forwards to PGW; PGW will allocate an IP address")

    e.run_step(Message(pgw, sgw, "Create Session Response", "control",
        {"ue_ip":     "10.0.0.42",
         "pgw_teid":  "0xDEADBEEF",
         "qos":       "APN-AMBR 50Mbps DL / 25Mbps UL"},
        interface="S5"),
        notes="PGW allocates IP address and GTP tunnel endpoint ID (TEID)")

    e.run_step(Message(sgw, mme, "Create Session Response", "control",
        {"sgw_teid": "0xCAFEBABE",
         "ue_ip":    "10.0.0.42"},
        interface="S11"),
        notes="SGW passes the response back to MME with its own TEID")

    # ------------------------------------------------------------------ #
    # Step 6: Attach Accept + RRC Reconfiguration
    # ------------------------------------------------------------------ #
    e.subsection("Phase 6: Attach Accept and Radio Bearer Setup")

    e.run_step(Message(mme, enb, "Initial Context Setup Request", "control",
        {"ue_ip":      "10.0.0.42",
         "sgw_teid":   "0xCAFEBABE",
         "erab_list":  "EPS Bearer ID 5",
         "ue_sec_cap": "AES, SNOW3G"},
        interface="S1-MME"),
        notes="MME tells eNB to set up the radio bearer and activate security")

    e.run_step(Message(enb, ue, "RRC Connection Reconfiguration", "control",
        {"drb_config":   "DRB1 (mapped to EPS Bearer 5)",
         "security_cfg": "AS ciphering activated"},
        interface="Uu"),
        notes="eNB sets up the Data Radio Bearer (DRB) for user traffic")

    e.run_step(Message(ue, enb, "RRC Connection Reconfiguration Complete", "control", {},
        interface="Uu"))

    e.run_step(Message(enb, mme, "Initial Context Setup Response", "control",
        {"enb_teid": "0xBEEFCAFE"},
        interface="S1-MME"),
        notes="eNB reports its GTP TEID so SGW knows where to send downlink data")

    e.run_step(Message(ue, mme, "Attach Complete", "control",
        {"eps_bearer_id": "5"},
        interface="NAS"),
        notes="Attach procedure is now complete")

    # ------------------------------------------------------------------ #
    # Step 7: User plane data flow
    # ------------------------------------------------------------------ #
    e.subsection("Phase 7: User Plane Data Flow (UE -> eNB -> SGW -> PGW -> Internet)")

    print("  State at end of attach:")
    print(f"    UE IP address : {e.state.get('ue_ip', 'N/A')}")
    print(f"    GTP tunnel    : eNB TEID=0xBEEFCAFE <-> SGW TEID=0xCAFEBABE")
    print(f"    QoS class     : QCI 9 (best-effort internet)")
    print()

    packets = [
        ("DNS query (UDP)   ",  "8.8.8.8:53",   43),
        ("HTTP GET          ", "93.184.216.34:80", 512),
        ("TLS Handshake     ", "93.184.216.34:443", 1024),
        ("Video stream chunk", "203.0.113.10:443",  1400),
    ]

    print("  Simulating uplink user plane packets:\n")
    print(f"  {'Packet type':22s} {'Destination':22s} {'Size (B)':10s} {'Path'}")
    print(f"  {'-'*22} {'-'*22} {'-'*10} {'-'*40}")
    for ptype, dst, size in packets:
        path = "UE -> eNB (Uu) -> SGW (S1-U/GTP) -> PGW (S5/GTP) -> Internet"
        print(f"  {ptype:22s} {dst:22s} {size:<10d} {path}")


# ---------------------------------------------------------------------------
# Architecture comparison table
# ---------------------------------------------------------------------------

def print_comparison_table():
    print()
    print("=" * 70)
    print("  Architecture Comparison: 2G -> 3G -> 4G -> 5G")
    print("=" * 70)

    headers = ["Feature", "2G (GSM)", "3G (UMTS)", "4G (LTE)", "5G NR"]
    rows = [
        ["Air interface",  "TDMA/FDMA",  "WCDMA",     "OFDMA",     "OFDM/NR"],
        ["RAN node",       "BTS",        "Node B",    "eNB",       "gNB (RU+DU+CU)"],
        ["RAN controller", "BSC",        "RNC",       "None (flat)", "CU (logical)"],
        ["Core type",      "CS + PS",    "CS + PS",   "All-IP EPC", "Cloud-native 5GC"],
        ["CP anchor",      "MSC/SGSN",   "SGSN",      "MME",       "AMF"],
        ["UP anchor",      "GGSN",       "GGSN",      "SGW/PGW",   "UPF"],
        ["Peak DL speed",  "~0.1 Mbps",  "~42 Mbps",  "~150 Mbps", "~20 Gbps"],
        ["Latency (RTT)",  "~300 ms",    "~100 ms",   "~30 ms",    "~1 ms (target)"],
        ["CP/UP split",    "No",         "No",        "Partial",   "Full (CUPS)"],
    ]

    col_w = [22, 12, 12, 14, 18]
    fmt = "  " + "".join(f"{{:<{w}}}" for w in col_w)

    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_w) + 4))
    for row in rows:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ProcedureEngine()
    run_lte_attach(engine)
    print_comparison_table()
    print()
    print("  Key takeaways:")
    print("  1. The attach procedure involves BOTH the control plane (NAS/RRC)")
    print("     and the user plane (GTP tunnels) being set up in sequence.")
    print("  2. The MME orchestrates authentication via the HSS and session")
    print("     setup via SGW/PGW -- it never touches user data directly.")
    print("  3. GTP tunnels carry user plane traffic through the core; each")
    print("     hop in the tunnel is identified by a TEID.")
    print("  4. In 5G, MME -> AMF, SGW -> UPF, and the gNB splits into RU/DU/CU.")
