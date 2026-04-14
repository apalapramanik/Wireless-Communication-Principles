"""
Topic 1 — Networking Fundamentals
UDP demo: connectionless, no reliability (video stream / game position scenario).

Simulates packet loss for frames 3 and 7 to show that UDP moves on
rather than stalling to retransmit.

Run:
    python udp_demo.py
"""

import socket
import threading
import time


HOST = "127.0.0.1"
PORT = 9101
TOTAL_FRAMES = 10
DROP_FRAMES   = {3, 7}    # simulate network drops


def udp_server(received_frames: list):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        s.settimeout(2.0)   # stop waiting after 2 s of silence

        while True:
            try:
                data, addr = s.recvfrom(128)
                frame = data.decode()
                received_frames.append(frame)
                print(f"[UDP Server] Received: {frame}")
            except socket.timeout:
                break


def udp_client():
    time.sleep(0.1)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for i in range(TOTAL_FRAMES):
            frame_msg = f"FRAME_{i:02d}  t={time.time():.3f}"

            if i in DROP_FRAMES:
                print(f"[UDP Client] FRAME_{i:02d} — LOST (simulated network drop) ❌")
                time.sleep(0.033)
                continue

            s.sendto(frame_msg.encode(), (HOST, PORT))
            print(f"[UDP Client] Sent:  {frame_msg}")
            time.sleep(0.033)   # ~30 fps cadence


if __name__ == "__main__":
    print("=" * 55)
    print("  UDP Demo — Video Stream (30 fps, 2 dropped frames)")
    print("=" * 55)
    print()

    received = []
    server_thread = threading.Thread(target=udp_server, args=(received,), daemon=True)
    server_thread.start()

    udp_client()
    server_thread.join(timeout=3)

    print()
    print(f"[UDP Client] Received {len(received)}/{TOTAL_FRAMES} frames")
    print(f"[UDP Client] Missing frames {sorted(DROP_FRAMES)} were just skipped — stream continued ⚡")
    print()
    print("  UDP trade-off: no retransmit, no stall.")
    print("  A tiny visual glitch beats a 200 ms freeze every time.")
