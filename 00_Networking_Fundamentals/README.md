# Topic 1 — Networking Fundamentals

Source: Kurose & Ross, Ch. 1, 3, 4 | Chat discussion

## Theory Covered
- What is a network? Hosts, links, switches, packets
- The TCP/IP protocol stack (5 layers) and encapsulation
- Circuit switching vs packet switching — why the internet chose packets
- The 4 sources of nodal delay: processing, queueing, transmission, propagation
- Traffic intensity `La/R` and why queuing delay explodes near saturation
- TCP: connection-oriented, 3-way handshake, reliability, flow/congestion control
- UDP: connectionless, no guarantees, low overhead
- Why 5G uses UDP (GTP-U) for the user plane tunnel

## Files

| File | What it demonstrates |
|------|----------------------|
| [delay_calculator.py](delay_calculator.py) | Compute all 4 delay components; plot queuing delay vs traffic intensity |
| [tcp_demo.py](tcp_demo.py) | TCP server + client (file download scenario, 3-way handshake) |
| [udp_demo.py](udp_demo.py) | UDP server + client (video stream scenario, simulated packet loss) |
| [encapsulation.py](encapsulation.py) | Layer-by-layer packet encapsulation and decapsulation visualizer |

## How to Run

```bash
cd 00_Networking_Fundamentals
pip install matplotlib numpy   # only needed for delay_calculator.py

python delay_calculator.py     # prints delay breakdown + saves traffic_intensity.png
python tcp_demo.py             # runs TCP server+client in-process
python udp_demo.py             # runs UDP server+client, shows dropped frames
python encapsulation.py        # prints packet structure at each layer
```

## Key Takeaways
- Queueing delay is the only one that blows up — all others are bounded
- TCP adds ~1.5 RTT overhead (handshake) but guarantees every byte arrives
- UDP saves that overhead at the cost of "fire and forget"
- Bandwidth (Hz) ≠ data rate (bps) — Shannon's law bridges them: `C = B × log2(1 + SNR)`
