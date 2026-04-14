"""
Topic 1 — Networking Fundamentals
Packet encapsulation visualizer — shows how each TCP/IP layer wraps
the layer above with its own header (and trailer at layer 2).

    Application  →  Transport  →  Network  →  Data Link  →  Physical

Run:
    python encapsulation.py
"""


def encapsulate(payload: str) -> dict[str, str]:
    """
    Simulate layer-by-layer encapsulation of an HTTP GET request.

    Returns an ordered dict: layer label → PDU string.
    """
    # Layer 5 — Application
    app = payload

    # Layer 4 — Transport: TCP header added
    tcp_hdr = "[TCP | src=52341 | dst=80 | seq=1000 | flags=SYN-ACK]"
    transport = f"{tcp_hdr}  {app}"

    # Layer 3 — Network: IP header added
    ip_hdr = "[IP | src=192.168.1.10 | dst=142.250.80.46 | TTL=64 | proto=TCP]"
    network = f"{ip_hdr}  {transport}"

    # Layer 2 — Data Link: Ethernet header + trailer
    eth_hdr     = "[ETH | src=AA:BB:CC:DD:EE:FF | dst=11:22:33:44:55:66 | type=IPv4]"
    eth_trailer = "[FCS]"
    datalink = f"{eth_hdr}  {network}  {eth_trailer}"

    # Layer 1 — Physical: bits on the wire (conceptual)
    physical = "10110010 01101110 00110101 ...  (NRZ encoded bits)"

    return {
        "Layer 5 — Application": app,
        "Layer 4 — Transport  ": transport,
        "Layer 3 — Network    ": network,
        "Layer 2 — Data Link  ": datalink,
        "Layer 1 — Physical   ": physical,
    }


def decapsulate(layers: dict[str, str]) -> None:
    """
    Show the reverse: each hop strips its own header and
    passes the inner PDU up (or re-wraps for the next hop).
    """
    names = list(layers.keys())
    print("\n=== Decapsulation — what each node reads ===\n")

    nodes = {
        "Your NIC (outbound)":  "Layer 1 — Physical   ",
        "First router":         "Layer 2 — Data Link  ",
        "Internet routers":     "Layer 3 — Network    ",
        "Destination server":   "Layer 4 — Transport  ",
        "Web server process":   "Layer 5 — Application",
    }

    for node, layer_key in nodes.items():
        layer_key = layer_key.rstrip()
        # find the matching key
        matched = next((k for k in names if k.strip() == layer_key.strip()), None)
        if matched:
            print(f"  {node:<25} reads up to {matched.strip()}")
    print()


if __name__ == "__main__":
    request = "GET / HTTP/1.1  Host: google.com"
    layers  = encapsulate(request)

    print("=== Packet Encapsulation (Application → Physical) ===\n")
    for label, pdu in layers.items():
        print(f"{label}:")
        print(f"    {pdu}\n")

    decapsulate(layers)

    print("Key rules:")
    print("  • Each layer adds a header (layer 2 also adds a trailer).")
    print("  • Routers only read up to layer 3 — they never see TCP or HTTP.")
    print("  • Only the destination server unwraps all the way to application data.")
    print("  • Each link between routers gets a fresh Ethernet frame (new src/dst MAC).")
