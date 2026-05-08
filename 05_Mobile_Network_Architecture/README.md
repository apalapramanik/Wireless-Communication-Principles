# 05 — Mobile Network Architecture (Phase 2 · Topic 1)

End-to-end simulation of the **4G LTE attach procedure** — the message exchange that happens when a phone powers on and joins the network, gets authenticated, and is given an IP address and a data tunnel to the internet.

## Files

| File | Purpose |
|------|---------|
| [`lte_attach.py`](lte_attach.py) | Replays the full attach procedure with annotated control-plane / user-plane traces |

## Run

```bash
python3 lte_attach.py
```

## What the simulator covers

| Phase | Procedure | Interfaces |
|-------|-----------|------------|
| 1 | RRC Connection Setup (UE ↔ eNB) | Uu |
| 2 | Initial UE Message (eNB → MME) | S1-MME |
| 3 | EPS-AKA Authentication (MME ↔ HSS ↔ UE) | S6a, NAS |
| 4 | NAS Security Mode Command | NAS |
| 5 | Default EPS Bearer setup (MME → SGW → PGW) | S11, S5 |
| 6 | Initial Context Setup + RRC Reconfiguration | S1-MME, Uu |
| 7 | User-plane data flow over GTP tunnels | S1-U, S5 |

## Network nodes modeled

| Node | Plane | Role |
|------|-------|------|
| UE | — | User Equipment (the phone) |
| eNB | RAN | LTE base station |
| MME | Core control | Mobility, attach, authentication orchestration |
| HSS | Core control | Subscriber database, AKA vectors |
| SGW | Core user | User-plane anchor inside the operator network |
| PGW | Core user | Internet gateway, IP allocation |

## Concepts demonstrated

- **Control plane vs user plane** — NAS/RRC messages set up state; GTP tunnels carry user data
- **GTP tunnel endpoints (TEIDs)** — how each hop in the tunnel is addressed
- **EPS-AKA** — challenge/response authentication using the SIM's secret key K
- **NAS security** — ciphering and integrity protection for signaling
- **EPS bearers** — how QoS classes (QCI) get mapped from the core down to the radio (DRB)

The script also prints a side-by-side **2G → 3G → 4G → 5G architecture comparison** showing how RAN nodes, core anchors, and CP/UP split evolved across generations.
